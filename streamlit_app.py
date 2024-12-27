import streamlit as st
from transformers import pipeline
import soundfile, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    # TODO
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor,tokenizer, model 

# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    # TODO
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return tokenizer, model

# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    # TODO
    processor,tokenizer, model = load_whisper_model()
    
    waveform, sampling_rate = torchaudio.load(uploaded_file)
    
    waveform = waveform.squeeze()
    if sampling_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sampling_rate, new_freq=16000)

    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    processor=processor,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    )


    transcription = pipe(waveform, batch_size=8)["text"]
    return transcription

# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    #TODO

    ner_results = ner_pipeline(text)
    st.write("DOIG")
    for entity in ner_results:
        st.write("DOIG")
        st.write(f"Entity: {entity['word']}, Type: {entity['entity_group']}")


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "İlhan Arda Yavuz"
    STUDENT_ID = "150220304"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # TODO
    # Fill here to create the streamlit application by using the functions you filled
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'], accept_multiple_files=False)
    if uploaded_file is not None:
        with open("/tmp/" + uploaded_file.name, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
    
        transcription = transcribe_audio("/tmp/" + uploaded_file.name)


        tokenizer, model = load_ner_model()
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        extract_entities(transcription, ner_pipeline)

    

if __name__ == "__main__":
    main()
