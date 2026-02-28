//! Whisper transcription using candle-transformers.
//!
//! This module implements audio transcription using the Whisper model via candle.

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use memvid_core::{TranscriptionResult, TranscriptionSegment, WhisperConfig};
use std::path::Path;
use tokenizers::Tokenizer;

/// Whisper model wrapper for transcription
pub struct WhisperTranscriber {
    model: Model,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
}

#[allow(dead_code)]
enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl WhisperTranscriber {
    /// Create a new WhisperTranscriber, downloading the model if needed
    pub fn new(config: &WhisperConfig) -> Result<Self> {
        // Use GPU if available: Metal (macOS) or CUDA (NVIDIA)
        let device = Self::select_device();
        tracing::info!(device = ?device, "Using device for Whisper");
        let model_id = match config.model_name.as_str() {
            "whisper-small-en" => "openai/whisper-small.en",
            "whisper-small" => "openai/whisper-small",
            "whisper-tiny.en" => "openai/whisper-tiny.en",
            "whisper-tiny" => "openai/whisper-tiny",
            "whisper-base.en" => "openai/whisper-base.en",
            "whisper-base" => "openai/whisper-base",
            "whisper-medium.en" => "openai/whisper-medium.en",
            "whisper-medium" => "openai/whisper-medium",
            "whisper-large-v3" => "openai/whisper-large-v3",
            other => other, // Allow direct model IDs
        };

        tracing::info!(model_id = model_id, "Loading Whisper model");

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        // Download model files
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let model_path = repo.get("model.safetensors")?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)?;
        let model_config: Config = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load mel filters
        let mel_bytes = match model_config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            n => bail!("Unsupported number of mel bins: {}", n),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
        };
        let model = Model::Normal(m::model::Whisper::load(&vb, model_config.clone())?);

        tracing::info!("Whisper model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            config: model_config,
            mel_filters,
            device,
        })
    }

    /// Select the best available device (GPU if available, otherwise CPU)
    fn select_device() -> Device {
        // Try Metal (macOS Apple Silicon / AMD)
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                tracing::info!("Metal GPU available");
                return device;
            }
        }

        // Try CUDA (NVIDIA GPUs)
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                tracing::info!("CUDA GPU available");
                return device;
            }
        }

        // Fallback to CPU
        tracing::info!("Using CPU (no GPU acceleration)");
        Device::Cpu
    }

    /// Transcribe an audio file
    pub fn transcribe_file(&mut self, path: &Path) -> Result<TranscriptionResult> {
        // Decode audio to PCM
        let (pcm_data, duration_secs) = memvid_core::decode_audio_file(path)?;

        // Check audio statistics
        let audio_min = pcm_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let audio_max = pcm_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let audio_mean = pcm_data.iter().sum::<f32>() / pcm_data.len() as f32;
        let audio_rms = (pcm_data.iter().map(|x| x * x).sum::<f32>() / pcm_data.len() as f32).sqrt();

        tracing::info!(
            duration = duration_secs,
            samples = pcm_data.len(),
            min = audio_min,
            max = audio_max,
            mean = audio_mean,
            rms = audio_rms,
            "Audio decoded"
        );

        self.transcribe_pcm(&pcm_data, duration_secs)
    }

    /// Transcribe PCM audio samples (16kHz mono f32)
    pub fn transcribe_pcm(&mut self, pcm_data: &[f32], duration_secs: f32) -> Result<TranscriptionResult> {
        // Whisper processes audio in 30-second chunks
        const CHUNK_LENGTH: usize = 30 * 16000; // 30 seconds at 16kHz
        const N_FRAMES: usize = 3000; // frames per chunk
        const SAMPLE_RATE: f32 = 16000.0;

        // Detect and trim leading silence
        let silence_threshold = 0.01; // RMS threshold for silence
        let window_size = 1600; // 100ms windows at 16kHz

        let start_sample = find_speech_start(pcm_data, silence_threshold, window_size);
        let end_sample = find_speech_end(pcm_data, silence_threshold, window_size);

        let trimmed_start = start_sample as f32 / SAMPLE_RATE;
        let trimmed_end = end_sample as f32 / SAMPLE_RATE;

        tracing::info!(
            start_sample = start_sample,
            end_sample = end_sample,
            trimmed_start_sec = trimmed_start,
            trimmed_end_sec = trimmed_end,
            original_duration = duration_secs,
            "Trimmed silence"
        );

        // Use trimmed audio
        let pcm_data = &pcm_data[start_sample..end_sample];
        let _trimmed_duration = pcm_data.len() as f32 / SAMPLE_RATE;

        let mut all_text = String::new();
        let mut segments = Vec::new();

        // Process audio in chunks
        let num_chunks = (pcm_data.len() + CHUNK_LENGTH - 1) / CHUNK_LENGTH;

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * CHUNK_LENGTH;
            let chunk_end = (chunk_start + CHUNK_LENGTH).min(pcm_data.len());
            let chunk = &pcm_data[chunk_start..chunk_end];

            // Adjust timestamps to account for trimmed silence
            let start_time = trimmed_start + chunk_start as f32 / SAMPLE_RATE;
            let end_time = trimmed_start + chunk_end as f32 / SAMPLE_RATE;

            tracing::info!(
                chunk = chunk_idx + 1,
                total = num_chunks,
                start = start_time,
                end = end_time,
                "Processing chunk"
            );

            // Reset decoder KV cache for each new chunk
            match &mut self.model {
                Model::Normal(m) => m.decoder.reset_kv_cache(),
                Model::Quantized(m) => m.decoder.reset_kv_cache(),
            }

            // Convert chunk to mel spectrogram
            let mel = audio::pcm_to_mel(&self.config, chunk, &self.mel_filters);
            let n_mels = self.config.num_mel_bins;
            let mel_len = mel.len();
            let n_frames = mel_len / n_mels;

            if chunk_idx == 0 {
                // Print config for debugging
                tracing::info!(
                    num_mel_bins = self.config.num_mel_bins,
                    max_source_positions = self.config.max_source_positions,
                    max_target_positions = self.config.max_target_positions,
                    "Model config"
                );

                // Mel statistics
                let mel_min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
                let mel_max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mel_mean = mel.iter().sum::<f32>() / mel.len() as f32;

                tracing::info!(
                    mel_len = mel_len,
                    n_mels = n_mels,
                    n_frames = n_frames,
                    chunk_samples = chunk.len(),
                    expected_frames = 3000,
                    mel_min = mel_min,
                    mel_max = mel_max,
                    mel_mean = mel_mean,
                    "Mel spectrogram computed"
                );
            }

            // Ensure we have exactly 3000 frames (pad or truncate)
            // NOTE: mel array from pcm_to_mel is stored as [mel_bin_0_all_frames, mel_bin_1_all_frames, ...]
            // So each mel bin has n_frames contiguous values: mel[bin * n_frames + frame]
            let mel = if n_frames < N_FRAMES {
                // Pad each mel bin's frames with zeros to reach N_FRAMES
                let mut padded = vec![0.0f32; n_mels * N_FRAMES];
                for bin in 0..n_mels {
                    let src_start = bin * n_frames;
                    let dst_start = bin * N_FRAMES;
                    padded[dst_start..dst_start + n_frames].copy_from_slice(&mel[src_start..src_start + n_frames]);
                }
                padded
            } else if n_frames > N_FRAMES {
                // Truncate each mel bin's frames to N_FRAMES
                let mut truncated = vec![0.0f32; n_mels * N_FRAMES];
                for bin in 0..n_mels {
                    let src_start = bin * n_frames;
                    let dst_start = bin * N_FRAMES;
                    truncated[dst_start..dst_start + N_FRAMES].copy_from_slice(&mel[src_start..src_start + N_FRAMES]);
                }
                truncated
            } else {
                mel
            };

            let mel = Tensor::from_vec(
                mel,
                (1, n_mels, N_FRAMES),
                &self.device,
            )?;

            if chunk_idx == 0 {
                let mel_shape = mel.shape();
                tracing::info!(
                    mel_shape = ?mel_shape,
                    "Mel tensor shape"
                );
            }

            // Run encoder
            let audio_features = match &mut self.model {
                Model::Normal(m) => m.encoder.forward(&mel, true)?,
                Model::Quantized(m) => m.encoder.forward(&mel, true)?,
            };

            if chunk_idx == 0 {
                let af_shape = audio_features.shape();
                tracing::info!(
                    audio_features_shape = ?af_shape,
                    "Audio features from encoder"
                );
            }

            // Get special token IDs
            let sot_token = self.token_id(m::SOT_TOKEN)?;
            let transcribe_token = self.token_id(m::TRANSCRIBE_TOKEN)?;
            let eot_token = self.token_id(m::EOT_TOKEN)?;
            let no_timestamps_token = self.token_id(m::NO_TIMESTAMPS_TOKEN)?;

            if chunk_idx == 0 {
                let en_token = self.tokenizer.token_to_id("<|en|>");
                tracing::info!(
                    sot = sot_token,
                    transcribe = transcribe_token,
                    eot = eot_token,
                    no_timestamps = no_timestamps_token,
                    en_token = ?en_token,
                    "Special tokens"
                );
            }

            // Build initial prompt
            // For English-only models (*.en), we DON'T use language token
            // For multilingual models, we add language token after sot_token
            let has_language_token = self.tokenizer.token_to_id("<|en|>").is_some();

            // English-only models have vocab size 51864, multilingual have 51865
            let is_english_only = self.config.vocab_size == 51864;

            let tokens = if is_english_only {
                // English-only: SOT -> transcribe -> notimestamps
                vec![sot_token, transcribe_token, no_timestamps_token]
            } else if has_language_token {
                // Multilingual: SOT -> language -> transcribe -> notimestamps
                let language_token = self.token_id("<|en|>")?;
                vec![sot_token, language_token, transcribe_token, no_timestamps_token]
            } else {
                // Fallback
                vec![sot_token, transcribe_token, no_timestamps_token]
            };

            if chunk_idx == 0 {
                tracing::info!(
                    is_english_only = is_english_only,
                    vocab_size = self.config.vocab_size,
                    prompt_tokens = ?tokens,
                    "Initial prompt"
                );
            }
            let mut all_tokens = tokens.clone();

            // Autoregressive decoding with token suppression
            let sample_len = self.config.max_target_positions / 2;
            let mut repeat_count = 0;
            let mut last_token: Option<u32> = None;

            // Build suppression mask
            let suppress_tokens = &self.config.suppress_tokens;

            for i in 0..sample_len {
                // For autoregressive decoding with KV cache:
                // - First iteration: pass all prompt tokens, flush_kv_cache=true
                // - Subsequent iterations: pass only the new token, flush_kv_cache=false
                let tokens_tensor = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

                if chunk_idx == 0 && i < 3 {
                    tracing::info!(
                        step = i,
                        all_tokens_len = all_tokens.len(),
                        tokens_shape = ?tokens_tensor.shape(),
                        "Decoder input"
                    );
                }

                // Get hidden states from decoder, then project to vocabulary
                // Always pass all tokens (candle doesn't use KV cache the same way as PyTorch)
                let logits = match &mut self.model {
                    Model::Normal(m) => {
                        let hidden = m.decoder.forward(&tokens_tensor, &audio_features, true)?;
                        m.decoder.final_linear(&hidden)?
                    }
                    Model::Quantized(m) => {
                        let hidden = m.decoder.forward(&tokens_tensor, &audio_features, true)?;
                        m.decoder.final_linear(&hidden)?
                    }
                };

                if chunk_idx == 0 && i == 0 {
                    tracing::info!(
                        logits_shape = ?logits.shape(),
                        "Decoder output logits"
                    );
                }

                // Get logits for last position
                let (_, seq_len, _) = logits.dims3()?;
                let mut logits = logits.i((0, seq_len - 1, ..))?.to_vec1::<f32>()?;

                // Apply token suppression from config
                for &token_id in suppress_tokens.iter() {
                    if (token_id as usize) < logits.len() {
                        logits[token_id as usize] = f32::NEG_INFINITY;
                    }
                }

                // Suppress EOT token for first few steps to allow generation
                if all_tokens.len() < 10 {
                    logits[eot_token as usize] = f32::NEG_INFINITY;
                }

                // Suppress all special tokens during generation:
                // - SOT (50257), language tokens (50258-50261), task tokens (50358-50359),
                // - no_timestamps (50362), and timestamp tokens (50363+)
                logits[sot_token as usize] = f32::NEG_INFINITY;
                logits[transcribe_token as usize] = f32::NEG_INFINITY;
                logits[no_timestamps_token as usize] = f32::NEG_INFINITY;
                // Suppress all tokens from 50257 onward (special tokens) except those in normal vocab
                for token_id in 50257..logits.len() {
                    logits[token_id] = f32::NEG_INFINITY;
                }

                if chunk_idx == 0 && i == 0 {
                    tracing::info!(
                        suppress_count = suppress_tokens.len(),
                        eot_suppressed = all_tokens.len() < 10,
                        "Applied token suppression"
                    );
                }

                // Find argmax
                let next_token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(eot_token);

                if chunk_idx == 0 && i < 5 {
                    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                    tracing::info!(
                        step = i,
                        next_token = next_token,
                        max_logit = max_logit,
                        min_logit = min_logit,
                        "Decoding step"
                    );
                }

                if next_token == eot_token || next_token >= self.config.vocab_size as u32 {
                    if chunk_idx == 0 && i < 5 {
                        tracing::info!(next_token = next_token, eot = eot_token, "Stopping: EOT or invalid token");
                    }
                    break;
                }

                // Check for excessive repetition (stop if same token repeats >3 times)
                if Some(next_token) == last_token {
                    repeat_count += 1;
                    if repeat_count > 3 {
                        tracing::debug!("Breaking due to token repetition");
                        break;
                    }
                } else {
                    repeat_count = 0;
                }
                last_token = Some(next_token);

                all_tokens.push(next_token);
            }

            // Decode tokens to text for this chunk
            let prompt_len = if is_english_only { 3 } else { 4 };

            if chunk_idx == 0 {
                tracing::info!(
                    prompt_tokens = ?&all_tokens[..prompt_len],
                    generated_tokens = ?&all_tokens[prompt_len..],
                    total = all_tokens.len(),
                    "Generated tokens for chunk"
                );
            }

            let chunk_text = self.tokenizer
                .decode(&all_tokens[prompt_len..], true) // Skip prompt tokens
                .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;

            let trimmed_text = chunk_text.trim();
            if !trimmed_text.is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(trimmed_text);

                segments.push(TranscriptionSegment {
                    start: start_time,
                    end: end_time,
                    text: trimmed_text.to_string(),
                });
            }
        }

        Ok(TranscriptionResult {
            text: all_text.trim().to_string(),
            language: "en".to_string(),
            duration_secs,
            segments,
        })
    }

    fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow!("Token '{}' not found in vocabulary", token))
    }
}

/// Find the sample index where speech starts (after leading silence)
fn find_speech_start(samples: &[f32], threshold: f32, window_size: usize) -> usize {
    for i in (0..samples.len()).step_by(window_size) {
        let end = (i + window_size).min(samples.len());
        let window = &samples[i..end];
        let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
        if rms > threshold {
            // Found speech, go back a bit to not cut off the start
            return i.saturating_sub(window_size);
        }
    }
    0 // No silence found, return start
}

/// Find the sample index where speech ends (before trailing silence)
fn find_speech_end(samples: &[f32], threshold: f32, window_size: usize) -> usize {
    for i in (0..samples.len()).rev().step_by(window_size) {
        let start = i.saturating_sub(window_size);
        let window = &samples[start..=i.min(samples.len() - 1)];
        let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
        if rms > threshold {
            // Found speech, go forward a bit to not cut off the end
            return (i + window_size).min(samples.len());
        }
    }
    samples.len() // No silence found, return end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_transcriber_creation() {
        let config = WhisperConfig::default();
        let _transcriber = WhisperTranscriber::new(&config).unwrap();
    }
}
