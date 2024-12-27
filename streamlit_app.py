import streamlit as st
from transformers import pipeline
import soundfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    # TODO
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return model

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
    model = load_whisper_model()

    waveform , sampling_rate = soundfile.read(uploaded_file)
    #waveform = torch.tensor(waveform).unsqueeze(0)

    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
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
    for entity in ner_results:
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}")


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

        st.write("/tmp/" + uploaded_file.name)
        transcription = transcribe_audio("/tmp/" + uploaded_file.name)
        txt = st.text(transcription)

    

if __name__ == "__main__":
    main()
