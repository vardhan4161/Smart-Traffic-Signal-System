import numpy as np
import sounddevice as sd
import threading
import time
from loguru import logger
from typing import Optional, Callable

class SirenDetector:
    """
    Detects emergency vehicle sirens using real-time audio analysis (FFT).
    Looks for high-energy rhythmic modulation in the 500Hz - 1500Hz range.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100, 
                 window_size: int = 2048, 
                 callback: Optional[Callable[[bool], None]] = None):
        self.fs = sample_rate
        self.n_fft = window_size
        self.callback = callback
        
        self.is_running = False
        self.siren_detected = False
        self._thread: Optional[threading.Thread] = None
        
        # Detection parameters
        self.freq_range = (500, 1500)  # Typical siren freq spectrum
        self.energy_threshold = 0.05    # Power threshold
        self.modulation_buffer = []     # Store peak freqs to detect modulation
        self.buffer_limit = 20          # ~1 second of audio history
        
    def start(self):
        """Start the audio listening thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._audio_stream_loop, daemon=True)
        self._thread.start()
        logger.info("SirenDetector started.")

    def stop(self):
        """Stop the audio listening thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("SirenDetector stopped.")

    def _audio_stream_loop(self):
        """Core loop capturing audio chunks and running FFT."""
        try:
            with sd.InputStream(samplerate=self.fs, channels=1, callback=self._process_audio):
                while self.is_running:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"SirenDetector Error: {e}")
            self.is_running = False

    def _process_audio(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            logger.debug(f"SD Status: {status}")
            
        # 1. Normalize and run FFT
        audio_chunk = indata[:, 0]
        if np.max(np.abs(audio_chunk)) < 0.001:  # Silence
            self._update_state(False)
            return

        fft_data = np.abs(np.fft.rfft(audio_chunk))
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.fs)
        
        # 2. Filter frequencies of interest
        mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        relevant_mags = fft_data[mask]
        
        if len(relevant_mags) == 0:
            self._update_state(False)
            return

        # 3. Detect Energy and Peak Frequency
        peak_idx = np.argmax(relevant_mags)
        peak_freq = freqs[mask][peak_idx]
        peak_mag = relevant_mags[peak_idx] / np.sum(fft_data) # Normalized power

        # 4. Detect Modulation (rhythmic frequency change)
        if peak_mag > self.energy_threshold:
            self.modulation_buffer.append(peak_freq)
            if len(self.modulation_buffer) > self.buffer_limit:
                self.modulation_buffer.pop(0)
            
            # If frequency is shifting significantly (modulation), it's likely a siren
            if len(self.modulation_buffer) >= 5:
                freq_std = np.std(self.modulation_buffer)
                if freq_std > 50: # Standard "Wail" has wide sweep
                    self._update_state(True)
                else:
                    self._update_state(False)
        else:
            self.modulation_buffer.clear()
            self._update_state(False)

    def _update_state(self, is_detected: bool):
        """Update detection state and trigger callback if changed."""
        if is_detected != self.siren_detected:
            self.siren_detected = is_detected
            logger.info(f"Siren Status Changed: {'DETECTED' if is_detected else 'CLEARED'}")
            if self.callback:
                self.callback(is_detected)

if __name__ == "__main__":
    # Test script
    def my_cb(detected):
        print("ALARM!" if detected else "SILENCE")
        
    sd_test = SirenDetector(callback=my_cb)
    sd_test.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sd_test.stop()
