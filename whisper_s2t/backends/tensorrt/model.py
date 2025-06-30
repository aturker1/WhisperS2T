import os
import tokenizers
import ctranslate2
import numpy as np

from .trt_model import WhisperTRT
from .tokenizer import Tokenizer, get_tokenizer
from .hf_utils import download_model
from .engine_builder import build_trt_engine, TRTBuilderConfig, load_trt_build_config


from .. import WhisperModel
from ...configs import *

from pathlib import Path

import torch
import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

from .whisper_utils import log_mel_spectrogram
import json
from collections import OrderedDict

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

FAST_ASR_OPTIONS = {
    "beam_size": 1,
    "best_of": 1, # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": 'tiny',
}


BEST_ASR_CONFIG = {
    "beam_size": 5,
    "best_of": 1, # Placeholder
    "patience": 2,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": 'tiny',
}




def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.n_mels = config['n_mels']
        self.dtype = config['dtype']
        self.num_languages = config['num_languages']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self,
                           mel,
                           mel_input_lengths,
                           encoder_downsampling_factor=2):
        if self.encoder_config['plugin_config']['remove_input_padding']:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)

        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = mel_input_lengths

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                       mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       mel_input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        encoder_output = outputs['encoder_output']
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor

        return encoder_output, encoder_output_lengths


class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config['plugin_config']['paged_kv_cache'] == True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_input_lengths,
                 eot_id,
                 max_new_tokens=444,
                 num_beams=1):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [batch_size, 1, encoder_max_input_length]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(end_id=eot_id,
                                         pad_id=eot_id,
                                         num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full((encoder_outputs.shape[0], ),
                                                 encoder_outputs.shape[1],
                                                 dtype=torch.int32,
                                                 device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lens)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):
    def __init__(self,
                 engine_dir,
                 debug_mode=False,
                 assets_dir=None,
                 use_py_session=False,
                 **model_kwargs):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config('encoder', engine_dir)
        decoder_config = read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        is_multilingual = (decoder_config['vocab_size'] >= 51865)
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(
            ), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name = "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(
            ), "gpt2.tiktoken file is not existed in assets_dir"
        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=self.num_languages,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]
        if use_py_session:
            self.encoder = WhisperEncoding(engine_dir)
            self.decoder = WhisperDecoding(engine_dir,
                                           runtime_mapping,
                                           debug_mode=debug_mode)
        else:
            json_config = GptJsonConfig.parse_file(engine_dir / 'decoder' /
                                                   'config.json')
            assert json_config.model_config.supports_inflight_batching
            runner_kwargs = dict(engine_dir=engine_dir,
                                 is_enc_dec=True,
                                 max_batch_size=32,
                                 max_input_len=3000,
                                 max_output_len=444,
                                 max_beam_width=1,
                                 debug_mode=debug_mode,
                                 kv_cache_free_gpu_memory_fraction=0.9)
            self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.use_py_session = use_py_session

    def process_batch(
            self,
            mel,
            prompt_id,
            num_beams=1,
            max_new_tokens=444, 
            **kwargs):
        
        mel = mel.type(torch.float16)

        mel_input_lengths = torch.full(
            (mel.shape[0],),
            mel.shape[2],
            dtype=torch.int32,
            device=mel.device,
        )
        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)
        if self.use_py_session:
            encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
                mel, mel_input_lengths)
            encoder_max_input_length = torch.max(encoder_output_lengths).item()
            output_ids = self.decoder.generate(decoder_input_ids,
                                               encoder_output,
                                               encoder_max_input_length,
                                               encoder_output_lengths,
                                               self.eot_id,
                                               max_new_tokens=max_new_tokens,
                                               num_beams=num_beams)
        else:
            with torch.no_grad():
                outputs = self.model_runner_cpp.generate(
                    batch_input_ids=decoder_input_ids,
                    encoder_input_features=mel.transpose(1, 2),
                    encoder_output_lengths=mel_input_lengths // 2,
                    max_new_tokens=max_new_tokens,
                    end_id=self.eot_id,
                    pad_id=self.eot_id,
                    num_beams=num_beams,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()
                output_ids = outputs['output_ids'].cpu().numpy().tolist()

        return output_ids


class WhisperModelTRT(WhisperModel):
    def __init__(self,
                 model_name_or_path: str,
                 cpu_threads=4,
                 num_workers=1,
                 device="cuda",
                 device_index=0,
                 compute_type="float16",
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 asr_options={},
                 **model_kwargs):

        # ASR Options
        self.asr_options = FAST_ASR_OPTIONS
        self.asr_options.update(asr_options)

        trt_build_args = TRTBuilderConfig(
            max_output_len=max_text_token_len, 
            max_beam_width=self.asr_options["beam_size"]
        )
        

        if 'trt_build_args' in model_kwargs:
            del model_kwargs['trt_build_args']

        self.trt_build_args = trt_build_args

        # Load model
        self.model = WhisperTRTLLM(model_name_or_path,assets_dir=model_name_or_path+"/../assets", **model_kwargs)
        
        # Load tokenizer
        # tokenizer_file = os.path.join(model_name_or_path, "tokenizer.json")
        tokenizer = Tokenizer(self.model.tokenizer, True)

        if self.asr_options['word_timestamps']:
            self.aligner_model_path = download_model(self.asr_options['word_aligner_model'])
            self.aligner_model = ctranslate2.models.Whisper(self.aligner_model_path,
                                                            device=device,
                                                            device_index=device_index,
                                                            compute_type=compute_type,
                                                            intra_threads=cpu_threads,
                                                            inter_threads=num_workers)
        
        self.generate_kwargs = {
            "end_id": tokenizer.eot,
            "pad_id": tokenizer.eot,
            "max_new_tokens": max_text_token_len,
            "length_penalty": self.asr_options['length_penalty'],
            "repetition_penalty": self.asr_options['repetition_penalty'],
            "num_beams": self.asr_options['beam_size'],
            "stop_words_list": self.asr_options['suppress_blank'],
            "bad_words_list": self.asr_options['suppress_tokens'],
            "temperature": self.asr_options['sampling_temperature'],
        }       

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if 'max_text_token_len' in params:
            self.update_params(params={'max_text_token_len': params['max_text_token_len']})
    
    def encode(self, features):
        """
        [Not Used]
        """
        
        return self.model.encode(features)

    def assign_word_timings(self, alignments, text_token_probs, words, word_tokens):
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])
    
        if len(word_tokens) <= 1:
            return []
            
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []
    
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps]*TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]
    
        return [
            dict(
                word=word, start=round(start, 2), end=round(end, 2), prob=round(prob, 2)
            )
            for word, start, end, prob in zip(
                words, start_times, end_times, word_probs
            )
        ]

    def align_words(self, features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata):
        lang_codes = [_['lang_code'] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(texts, text_tokens, lang_codes)

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                # print(_sot_seq)
                start_seq_wise_req[_sot_seq].append(_idx)
            except:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            res = self.aligner_model.align(ctranslate2.StorageView.from_array(features[req_idx]), 
                                           start_sequence=list(start_seq), 
                                           text_tokens=[text_tokens[_] for _ in req_idx],
                                           num_frames=list(seq_lens[req_idx].detach().cpu().numpy()), 
                                           median_filter_width=7)

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            _word_timings = self.assign_word_timings(token_alignments[_idx].alignments, 
                                                     token_alignments[_idx].text_token_probs, 
                                                     word_tokens[_idx][0], 
                                                     word_tokens[_idx][1])
        
            stitched_seg = _seg_metadata['stitched_seg']

            current_seg_idx = 0
            current_offset = _seg_metadata['start_time']
        
            for w in _word_timings:
                while (w['start'] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (stitched_seg[current_seg_idx][0]-stitched_seg[current_seg_idx-1][1])
        
                w['start'] += current_offset
                w['end'] += current_offset
        
            word_timings.append(_word_timings)

        return word_timings
    
    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata):

        result = self.model.process_batch(features, prompt_id=prompts[0], **self.generate_kwargs)
        
        texts = self.tokenizer.decode_batch([x[0] for x in result])
        
        response = []
        for idx, r in enumerate(result):
            response.append({'text': texts[idx].strip()})

        if self.asr_options['word_timestamps']:
            text_tokens = [[_t for _t in x[0] if _t < self.tokenizer.eot]+[self.tokenizer.eot] for x in result]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            word_timings = self.align_words(features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata)

            for _response, _word_timings in zip(response, word_timings):
                _response['word_timestamps'] = _word_timings

        return response