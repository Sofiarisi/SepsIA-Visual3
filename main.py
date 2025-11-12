import os
import json
import re # Usaremos regex para analizar la respuesta de Gemini
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

# --- 1. Configuración Inicial ---

# Carga la variable de entorno (GOOGLE_API_KEY) desde el archivo .env
load_dotenv()

# Inicializa la API de FastAPI
app = FastAPI(
    title="API de Asistente de Decisión Clínica (Sepsis)",
    description="Una API que analiza datos de pacientes para detectar riesgo de sepsis usando Gemini.",
    version="1.0.0"
)

# Configura la API Key de Gemini desde la variable de entorno
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No se encontró la GOOGLE_API_KEY en las variables de entorno.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error al configurar Gemini: {e}")
    # En un entorno de producción, esto debería detener el inicio de la app
    
# --- 2. Definición de Modelos de Datos (Pydantic) ---
# Esto valida automáticamente los datos que llegan a tu API

class PatientData(BaseModel):
    frecuencia_cardiaca: int = Field(..., example=125)
    presion_arterial: str = Field(..., example="85/45")
    frecuencia_respiratoria: int = Field(..., example=28)
    temperatura_corporal: float = Field(..., example=39.1)
    saturacion_oxigeno: int = Field(..., example=89)
    procalcitonina: float = Field(..., example=4.5)
    lactato: float = Field(..., example=3.1)
    pcr: float = Field(..., example=210.0)
    leucocitos: float = Field(..., example=18.2)
    array_json_comorbilidades: list[str] = Field(..., example=["EPOC", "Hipertensión"])
    texto_patologias_presentes: str = Field(..., example="Neumonía Adquirida en la Comunidad (NAC)")
    texto_sintomas_diarios: str = Field(..., example="Confusión aguda, disnea severa.")
    objeto_json_valores_previos: dict = Field(..., example={"lactato_previo": 1.1, "pa_previa": "110/70"})
    porcentaje_ia: int = Field(..., example=85)

class AnalysisResponse(BaseModel):
    analisis_breve: str
    justificacion: str
    acciones_sugeridas: str

# --- 3. Lógica de Gemini---

# La plantilla de prompt
prompt_base = """### ROL Y OBJETIVO ###
Actúa como un médico experto en cuidados intensivos y soporte de decisiones clínicas, especializado en la detección temprana de sepsis. Tu propósito es asistir a médicos y enfermeras calificados en un entorno de paciente internado.

### CONTEXTO ###
Estás analizando los datos de un paciente para identificar el riesgo de sepsis o shock séptico. Un modelo de IA interno (de la aplicación) ha proporcionado una puntuación de riesgo preliminar. Tu tarea es analizar la totalidad de los datos (clínicos, laboratorio, tendencias y comorbilidades) para proveer un análisis accionable e integrado.

### DATOS DEL PACIENTE ###
Analiza los siguientes datos. Presta especial atención a las combinaciones y tendencias que sugieran disfunción orgánica múltiple.

**1. Datos Actuales (Signos Vitales y Laboratorio Estándar):**
* Frecuencia Cardíaca: {FRECUENCIA_CARDIACA} (lat/min)
* Presión Arterial (Sistólica/Media): {PRESION_ARTERIAL} (mmHg)
* Frecuencia Respiratoria: {FRECUENCIA_RESPIRATORIA} (resp/min)
* Temperatura Corporal: {TEMPERATURA_CORPORAL} (°C)
* Saturación de Oxígeno (SpO2): {SATURACION_OXIGENO} (%)
* Procalcitonina (PCT): {PROCALCITONINA} (ng/mL)
* Lactato: {LACTATO} (mmol/L)
* Proteína C Reactiva (PCR): {PCR} (mg/L)
* Leucocitos: {LEUCOCITOS} (x10^9/L)

**2. Historial y Contexto (Formato Estructurado):**
* Comorbilidades Conocidas (Array JSON): {ARRAY_JSON_COMORBILIDADES}
* Patologías Presentes (String): "{TEXTO_PATOLOGIAS_PRESENTES}"
* Síntomas Diarios Reportados (String): "{TEXTO_SINTOMAS_DIARIOS}"

**3. Datos de Tendencia y Modelo Interno:**
* Valores Previos (Objeto JSON): {OBJETO_JSON_VALORES_PREVIOS}
* Puntuación de Riesgo de Sepsis (Modelo IA Interno): {PORCENTAJE_IA}%

### INSTRUCCIONES DE ANÁLIS... (etc.) ...###

### FORMATO DE RESPUESTA OBLIGATORIO ###
Proporciona tu respuesta estrictamente en el siguiente formato de 3 secciones, respetando los límites de líneas:

**1. Análisis Breve:**
(Resumen conciso del estado y riesgo. **MÁXIMO 2 LÍNEAS.**)

**2. Justificación:**
(Explicación concisa. Enfocarse *solo* en los 2-3 factores críticos (ej. "Hipotensión + Lactato elevado") que justifican el análisis. **MÁXIMO 5 LÍNEAS.**)

**3. Acciones Sugeridas:**
(Lista de 3-4 acciones *más urgentes* y priorizadas. Sin explicaciones. **MÁXIMO 5 LÍNEAS en total para esta sección.**)
"""

# Configuraciones de Gemini
safety_settings_config = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
generation_config = genai.GenerationConfig(
    temperature=0.2,
    max_output_tokens=4096 
)

# Inicializa el modelo
model = genai.GenerativeModel('gemini-2.0-flash')

# --- 4. Parseo ---

def parse_gemini_response(text: str) -> dict:
    """
    Analiza la respuesta de texto de Gemini y la divide en un dict estructurado.
    """
    try:
        # Usamos regex para encontrar el contenido después de cada título
        # re.DOTALL significa que . también incluye saltos de línea
        
        analisis = re.search(r"\*\*1\. Análisis Breve:\*\*(.*?)(?=\*\*2\. Justificación:\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
        justificacion = re.search(r"\*\*2\. Justificación:\*\*(.*?)(?=\*\*3\. Acciones Sugeridas:\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
        acciones = re.search(r"\*\*3\. Acciones Sugeridas:\*\*(.*)", text, re.DOTALL | re.IGNORECASE)

        # Limpiamos el texto (quitamos espacios, saltos de línea y texto de placeholder)
        clean_analisis = analisis.group(1).strip().strip("()").strip() if analisis else "No se pudo parsear el análisis."
        clean_justificacion = justificacion.group(1).strip().strip("()").strip() if justificacion else "No se pudo parsear la justificación."
        clean_acciones = acciones.group(1).strip().strip("()").strip() if acciones else "No se pudieron parsear las acciones."

        return {
            "analisis_breve": clean_analisis,
            "justificacion": clean_justificacion,
            "acciones_sugeridas": clean_acciones
        }
    except Exception as e:
        print(f"Error al parsear la respuesta: {e}")
        # Devuelve el texto en bruto si falla el parseo
        return {
            "analisis_breve": text,
            "justificacion": "Error de parseo.",
            "acciones_sugeridas": "Error de parseo."
        }


# --- 5. El Endpoint de la API ---

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient_endpoint(data: PatientData):
    """
    Recibe los datos de un paciente, construye el prompt,
    llama a la API de Gemini y devuelve un análisis estructurado.
    """
    try:
        # Formatear las variables estructuradas
        comorbilidades_formateada = json.dumps(data.array_json_comorbilidades, indent=2, ensure_ascii=False)
        valores_previos_formateados = json.dumps(data.objeto_json_valores_previos, indent=2, ensure_ascii=False)

        # Rellenar la plantilla del prompt
        prompt_final = prompt_base.format(
            FRECUENCIA_CARDIACA=data.frecuencia_cardiaca,
            PRESION_ARTERIAL=data.presion_arterial,
            FRECUENCIA_RESPIRATORIA=data.frecuencia_respiratoria,
            TEMPERATURA_CORPORAL=data.temperatura_corporal,
            SATURACION_OXIGENO=data.saturacion_oxigeno,
            PROCALCITONINA=data.procalcitonina,
            LACTATO=data.lactato,
            PCR=data.pcr,
            LEUCOCITOS=data.leucocitos,
            ARRAY_JSON_COMORBILIDADES=comorbilidades_formateada,
            TEXTO_PATOLOGIAS_PRESENTES=data.texto_patologias_presentes,
            TEXTO_SINTOMAS_DIARIOS=data.texto_sintomas_diarios,
            OBJETO_JSON_VALORES_PREVIOS=valores_previos_formateados,
            PORCENTAJE_IA=data.porcentaje_ia
        )

        # Llamar a Gemini
        response = model.generate_content(
            prompt_final,
            generation_config=generation_config,
            safety_settings=safety_settings_config
        )

        # Analizar (parsear) la respuesta
        parsed_response = parse_gemini_response(response.text)
        
        return parsed_response

    except Exception as e:
        print(f"Error en el endpoint /analyze_patient: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Análisis de Sepsis. Ve a /docs para probar."}