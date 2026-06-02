import os
import sys
import numpy as np
import sounddevice as sd
import sherpa_onnx

def main():
    # Ruta a la carpeta del modelo que descargamos antes
    model_dir = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-04-13"
    
    if not os.path.exists(model_dir):
        print(f"Error: No se encuentra la carpeta del modelo '{model_dir}'")
        print("Asegúrate de haberlo descargado y descomprimido en esta misma carpeta.")
        return

    # Configuración del motor de búsqueda de palabras clave
    config = sherpa_onnx.KeywordSpotterConfig(
        feat_config=sherpa_onnx.FeatureConfig(sample_rate=16000, feature_dim=80),
        model_config=sherpa_onnx.KeywordSpotterModelConfig(
            transducer=sherpa_onnx.OnlineTransducerModelConfig(
                encoder=os.path.join(model_dir, "encoder.onnx"),
                decoder=os.path.join(model_dir, "decoder.onnx"),
                joiner=os.path.join(model_dir, "joiner.onnx"),
            ),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=4, # Aprovechamos los 4 núcleos de la potente Raspberry Pi 5
            provider="cpu",
        ),
        keywords_file=os.path.join(model_dir, "keywords.txt"),
    )

    if not config.validate():
        print("Error: Los archivos del modelo ONNX no son válidos.")
        return

    print("🧠 Cargando modelo de Inteligencia Artificial...")
    kws = sherpa_onnx.KeywordSpotter(config=config)
    stream = kws.create_stream()

    sample_rate = 16000
    block_size = int(0.1 * sample_rate) # Procesamos el audio en bloques de 100ms

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        
        # Convertir el audio del micrófono USB a flotantes de 32 bits
        samples = indata[:, 0].astype(np.float32)
        stream.accept_waveform(sample_rate, samples)
        
        # Procesar y decodificar el audio recibido
        while kws.is_ready(stream):
            kws.decode(stream)
        
        # Verificar si la palabra clave ha saltado
        result = kws.get_result(stream)
        if result.keyword:
            print("\n" + "🎉" * 15)
            print(f"¡PALABRA DETECTADA!: {result.keyword}")
            print("🎉" * 15 + "\n")
            
            # Reseteamos el stream para que vuelva a escuchar la siguiente palabra
            kws.input_finished(stream)

    # Abrir el flujo del micrófono USB
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate, blocksize=block_size, callback=audio_callback):
        print("\n🎙️  Micrófono activo y escuchando de fondo...")
        print("Prueba a decir claramente: 'buenos días' o 'hola de nuevo'")
        print("Presiona Ctrl+C para detener el programa.\n")
        while True:
            sd.sleep(100)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Detector apagado correctamente.")    