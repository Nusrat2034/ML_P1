import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_normalize_audio(file_path, sr=22050):
    """Load audio file and normalize it"""
    audio, _ = librosa.load(file_path, sr=sr)
    # Normalize to [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio

def match_length(audio, noise):
    """Make audio and noise the same length"""
    len_audio = len(audio)
    len_noise = len(noise)
    
    if len_audio > len_noise:
        # Repeat noise to match audio length
        repeat_times = int(np.ceil(len_audio / len_noise))
        noise = np.tile(noise, repeat_times)[:len_audio]
    elif len_noise > len_audio:
        # Truncate noise to match audio length
        noise = noise[:len_audio]
    
    return audio, noise

def mix_audio_with_noise(audio, noise, snr_db=10):
    """Mix audio with noise at specified SNR (Signal-to-Noise Ratio)"""
    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate desired noise power based on SNR
    snr_linear = 10 ** (snr_db / 10)
    desired_noise_power = signal_power / snr_linear
    
    # Scale noise
    if noise_power > 0:
        noise_scaled = noise * np.sqrt(desired_noise_power / noise_power)
    else:
        noise_scaled = noise
    
    # Mix audio and noise
    mixed = audio + noise_scaled
    
    # Normalize mixed audio to prevent clipping
    if np.max(np.abs(mixed)) > 0:
        mixed = mixed / np.max(np.abs(mixed)) * 0.9
    
    return mixed

def generate_spectrogram(audio, sr, save_path, title):
    """Generate and save spectrogram"""
    plt.figure(figsize=(10, 4))
    
    # Compute spectrogram
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Display spectrogram
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_audio_mixing(data_dir='data', snr_db=10):
    """Main function to process audio mixing"""
    
    # Define paths
    audio_dir = Path(data_dir) / 'audio'
    noise_dir = Path(data_dir) / 'noise'
    mixed_dir = Path(data_dir) / 'mixed'
    spectrogram_dir =  Path(data_dir) / 'spectrograms'
    
    # Create output directories
    mixed_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    
    # Get audio and noise files
    audio_files = sorted(list(audio_dir.glob('*.wav')))
    noise_files = sorted(list(noise_dir.glob('*.wav')))
    
    if not audio_files:
        print("No audio files found in audio directory!")
        return
    
    if not noise_files:
        print("No noise files found in noise directory!")
        return
    
    print(f"Found {len(audio_files)} audio files and {len(noise_files)} noise files")
    print(f"SNR: {snr_db} dB\n")
    
    # Process each audio file with each noise file
    for i, audio_file in enumerate(audio_files):
        for j, noise_file in enumerate(noise_files):
            print(f"Processing: {audio_file.name} + {noise_file.name}")
            
            # Load audio and noise
            audio = load_and_normalize_audio(str(audio_file))
            noise = load_and_normalize_audio(str(noise_file))
            
            # Match lengths
            audio, noise = match_length(audio, noise)
            
            # Mix audio with noise
            mixed = mix_audio_with_noise(audio, noise, snr_db=snr_db)
            
            # Create output filename
            audio_name = audio_file.stem
            noise_name = noise_file.stem
            output_name = f"{audio_name}_+_{noise_name}"
            
            # Save mixed audio
            mixed_path = mixed_dir / f"{output_name}.wav"
            sf.write(str(mixed_path), mixed, 22050)
            print(f"  Saved: {mixed_path}")
            
            # Generate and save spectrogram
            spec_path = spectrogram_dir / f"{output_name}_spectrogram.png"
            generate_spectrogram(mixed, 22050, str(spec_path), 
                               f"Spectrogram: {audio_name} + {noise_name}")
            print(f"  Saved: {spec_path}\n")
    
    print("Processing complete!")
    print(f"Mixed audio files saved in: {mixed_dir}")
    print(f"Spectrograms saved in: {spectrogram_dir}")

if __name__ == "__main__":
    # Run the mixing process with SNR of 10 dB
    # You can adjust the SNR value (higher = less noise, lower = more noise)
    process_audio_mixing(data_dir='data', snr_db=10)