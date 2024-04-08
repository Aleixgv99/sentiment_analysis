import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLabel
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Función para mostrar el indicador de sentimiento con Plotly
# Función para mostrar el indicador de sentimiento con Plotly
import plotly.graph_objects as go

def show_sentiment_indicator(sentiment_value):
    sentiment_value_scaled = sentiment_value * 100  # Escala al rango [0, 100]
    
    # Define los colores para el degradado
    gauge_background_colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#FFFF00", "#ADFF2F", "#9ACD32", "#008000"]
    
    # Crea los 'steps' para el degradado del medidor
    steps = []
    num_steps = len(gauge_background_colors)
    for i, color in enumerate(gauge_background_colors):
        steps.append({'range': [i * (100/num_steps), (i+1) * (100/num_steps)], 'color': color})

    # Define el indicador
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_value_scaled,
        domain={'x': [0, 1], 'y': [0.1, 0.8]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black", 'thickness': 0.02},
            'bgcolor': "white",
            'steps': steps,
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_value_scaled
            }
        },
    ))
    
    # Mueve el título y el valor del número a la parte superior
    fig.update_layout(
        title={'text': "Sentiment Analysis", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top','font': {'size': 60}},
        paper_bgcolor="white",
        font={'color': "black", 'family': "Arial"},
        margin={'t': 20, 'b': 0, 'l': 0, 'r': 0},  # Reduce el margen para que el título encaje
    )
    
    # Ajusta la posición del valor del número
    fig.update_traces(number={'valueformat': '.1f', 'suffix': None, 'font': {'size': 40}})
    
    fig.show()

class SentimentAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Análisis de Sentimiento de Criptomonedas')
        self.setGeometry(100, 100, 600, 400)

        # Cargar el modelo
        self.model = load_model('/home/developer/Documents/Master/TFM/models/model_lstm/')  # Asegúrate de que la ruta es correcta
        
        # Inicializar el tokenizador y el lematizador
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        
        self.words = self.load_glove_vectors('/home/developer/Documents/Master/TFM/Glove/glove.6B.50d.txt')  # Reemplaza con la ruta real

        self.initUI()

    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout()

        self.textEdit = QTextEdit()
        self.textEdit.setPlaceholderText("Introduce tu texto aquí...")
        self.layout.addWidget(self.textEdit)

        self.analyzeButton = QPushButton("Analizar Sentimiento")
        self.analyzeButton.clicked.connect(self.analyzeSentiment)
        self.layout.addWidget(self.analyzeButton)

        self.resultLabel = QLabel("Resultado del análisis aparecerá aquí")
        self.layout.addWidget(self.resultLabel)

        self.centralWidget.setLayout(self.layout)
        
    def load_glove_vectors(self, filename):
        word_vectors = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float32)
                word_vectors[word] = embedding
        return word_vectors
        
    def message_to_word_vectors(self, message):
        tokens = self.tokenizer.tokenize(message.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        vectors = [self.words.get(t, np.zeros((50,))) for t in lemmatized_tokens]
        return np.array(vectors)

    def analyzeSentiment(self):
            user_input_text = self.textEdit.toPlainText()
            word_vectors = self.message_to_word_vectors(user_input_text)

            # Si tu modelo espera una secuencia fija y no secuencias de longitud variable:
            # Asegúrate de que 'max_length' sea la longitud máxima de secuencia que tu modelo puede manejar.
            max_length = 60  # o cualquier otra longitud que hayas usado durante el entrenamiento
            if len(word_vectors) > max_length:
                word_vectors = word_vectors[:max_length]
            elif len(word_vectors) < max_length:
                padding = np.zeros((max_length - len(word_vectors), 50))
                word_vectors = np.vstack((word_vectors, padding))

            # Realizar la predicción
            prediction = self.model.predict(word_vectors[np.newaxis, ...])
            pred = float(prediction)
            show_sentiment_indicator(pred)

            # Interpretar la salida del modelo
            sentiment_result = "Positivo" if prediction[0] > 0.5 else "Negativo"
            self.resultLabel.setText(f"Sentimiento del análisis: {sentiment_result}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = SentimentAnalysisApp()
    mainWindow.show()
    sys.exit(app.exec_())
