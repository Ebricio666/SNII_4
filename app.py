from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SNII Insight",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.02);
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }

        .snii-subtitle {
            color: #746a80;
            margin-top: -0.6rem;
            margin-bottom: 1.1rem;
        }

        .snii-note {
            border-left: 4px solid #6d28d9;
            padding: 0.75rem 1rem;
            background: rgba(109, 40, 217, 0.08);
            border-radius: 0 12px 12px 0;
            margin: 0.5rem 0 1rem;
        }

        .lab-chip {
            display: inline-block;
            padding: 0.30rem 0.70rem;
            margin: 0.15rem 0.20rem;
            border-radius: 999px;
            font-weight: 650;
            border: 1px solid transparent;
        }

        .lab-variable {
            background: rgba(37, 99, 235, 0.12);
            border-color: rgba(37, 99, 235, 0.35);
            color: #1d4ed8;
        }

        .lab-tiempo {
            background: rgba(22, 163, 74, 0.12);
            border-color: rgba(22, 163, 74, 0.35);
            color: #15803d;
        }

        .lab-filtro {
            background: rgba(234, 88, 12, 0.12);
            border-color: rgba(234, 88, 12, 0.35);
            color: #c2410c;
        }

        .lab-ubicacion {
            background: rgba(124, 58, 237, 0.12);
            border-color: rgba(124, 58, 237, 0.35);
            color: #6d28d9;
        }

        .lab-objetivo {
            background: rgba(219, 39, 119, 0.12);
            border-color: rgba(219, 39, 119, 0.35);
            color: #be185d;
        }

        .lab-sentence {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.7rem 0 1rem;
            font-size: 1.02rem;
            line-height: 2.1;
        }

        .lab-score {
            font-size: 1.7rem;
            font-weight: 750;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTES
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    APP_DIR / "data" / "SNII_MASTER_v1_PERSONA_ANIO.parquet",
    APP_DIR / "SNII_MASTER_v1_PERSONA_ANIO.parquet",
    APP_DIR / "data" / "SNII_MASTER_v1_PERSONA_ANIO.xlsx",
    APP_DIR / "SNII_MASTER_v1_PERSONA_ANIO.xlsx",
]

COLUMNAS_MINIMAS = [
    "ID_PERSONA_EXACTA",
    "AÑO",
]

COLUMNAS_ANALITICAS = [
    "ID_PERSONA_EXACTA",
    "AÑO",
    "CVU_REFERENCIA",
    "NOMBRE_INVESTIGADOR",
    "SEXO_CONSOLIDADO",
    "PRIMER_AÑO",
    "ULTIMO_AÑO",
    "NUMERO_AÑOS_PRESENTE",
    "ESTA_VIGENTE_EN_2025",
    "INSTITUCION_ANUAL",
    "DEPENDENCIA_ANUAL",
    "SUBDEPENDENCIA_ANUAL",
    "ENTIDAD_FEDERATIVA_ANUAL",
    "PAIS_ANUAL",
    "NIVEL_SNII_STD",
    "NIVEL_SNII_ETIQUETA",
    "AREA_DEL_CONOCIMIENTO_ANUAL",
    "CAMPO_DEL_CONOCIMIENTO_ANUAL",
    "DISCIPLINA_ANUAL",
    "SUBDISCIPLINA_ANUAL",
    "ESPECIALIDAD_ANUAL",
    "CLASIFICACION_STEM_ANUAL",
    "GRUPO_STEM_BINARIO",
    "ES_STEM_ESTRICTO",
    "ES_STEM_AMPLIADO",
    "PORCENTAJE_COMPLETITUD_CLAVE",
    "REQUIERE_REVISION_MASTER",
]


# ============================================================
# UTILIDADES DE DATOS
# ============================================================

def limpiar_texto(serie: pd.Series) -> pd.Series:
    """Normaliza valores textuales vacíos sin destruir el dato."""
    resultado = serie.astype("string").str.strip()

    marcadores = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "NA",
        "N/A",
        "NO HAY INFORMACIÓN AL RESPECTO",
        "NO HAY INFORMACION AL RESPECTO",
    }

    return resultado.mask(resultado.str.upper().isin(marcadores))


def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara tipos esenciales y conserva sólo columnas disponibles."""
    faltantes = [col for col in COLUMNAS_MINIMAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            "La base no contiene las columnas indispensables: "
            + ", ".join(faltantes)
        )

    df = df.copy()

    df["ID_PERSONA_EXACTA"] = limpiar_texto(df["ID_PERSONA_EXACTA"])
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")

    columnas_texto = [
        col
        for col in [
            "CVU_REFERENCIA",
            "NOMBRE_INVESTIGADOR",
            "SEXO_CONSOLIDADO",
            "INSTITUCION_ANUAL",
            "DEPENDENCIA_ANUAL",
            "SUBDEPENDENCIA_ANUAL",
            "ENTIDAD_FEDERATIVA_ANUAL",
            "PAIS_ANUAL",
            "NIVEL_SNII_ETIQUETA",
            "AREA_DEL_CONOCIMIENTO_ANUAL",
            "CAMPO_DEL_CONOCIMIENTO_ANUAL",
            "DISCIPLINA_ANUAL",
            "SUBDISCIPLINA_ANUAL",
            "ESPECIALIDAD_ANUAL",
            "CLASIFICACION_STEM_ANUAL",
            "GRUPO_STEM_BINARIO",
        ]
        if col in df.columns
    ]

    for columna in columnas_texto:
        df[columna] = limpiar_texto(df[columna])

    if "NIVEL_SNII_STD" in df.columns:
        df["NIVEL_SNII_STD"] = pd.to_numeric(
            df["NIVEL_SNII_STD"],
            errors="coerce",
        ).astype("Int64")

    df = df.loc[
        df["ID_PERSONA_EXACTA"].notna()
        & df["AÑO"].between(2000, 2025, inclusive="both")
    ].copy()

    df = df.drop_duplicates(
        subset=["ID_PERSONA_EXACTA", "AÑO"],
        keep="last",
    )

    return df


@st.cache_data(show_spinner="Cargando la base histórica del SNII…")
def cargar_desde_ruta(ruta: str) -> pd.DataFrame:
    """Carga Parquet o Excel desde el repositorio."""
    path = Path(ruta)

    if path.suffix.lower() == ".parquet":
        # Se cargan todas las columnas procesadas del master.
        df = pd.read_parquet(
            path,
            engine="pyarrow",
        )

    elif path.suffix.lower() in {".xlsx", ".xls"}:
        # El Excel se conserva como respaldo; Parquet sigue siendo recomendado.
        df = pd.read_excel(
            path,
            sheet_name="PERSONA_AÑO",
            engine="openpyxl",
        )

    else:
        raise ValueError("Formato de archivo no compatible.")

    return preparar_base(df)


@st.cache_data(show_spinner="Procesando el archivo cargado…")
def cargar_desde_upload(
    contenido: bytes,
    nombre: str,
) -> pd.DataFrame:
    """Carga un archivo proporcionado desde la interfaz."""
    from io import BytesIO

    buffer = BytesIO(contenido)
    extension = Path(nombre).suffix.lower()

    if extension == ".parquet":
        df = pd.read_parquet(buffer, engine="pyarrow")

    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(
            buffer,
            sheet_name="PERSONA_AÑO",
            engine="openpyxl",
        )

    else:
        raise ValueError("Carga un archivo .parquet o .xlsx.")

    # Se conservan todas las columnas disponibles del archivo cargado.
    return preparar_base(df.copy())


def localizar_archivo() -> Path | None:
    for ruta in DATA_CANDIDATES:
        if ruta.exists():
            return ruta
    return None


def obtener_base() -> tuple[pd.DataFrame, str]:
    """Localiza la base del repositorio o solicita una carga manual."""
    ruta = localizar_archivo()

    if ruta is not None:
        return cargar_desde_ruta(str(ruta)), ruta.name

    st.sidebar.warning(
        "No se encontró el master dentro del repositorio. "
        "Carga temporalmente un archivo para probar la aplicación."
    )

    archivo = st.sidebar.file_uploader(
        "Cargar master",
        type=["parquet", "xlsx"],
    )

    if archivo is None:
        st.info(
            "Coloca `SNII_MASTER_v1_PERSONA_ANIO.parquet` dentro de "
            "la carpeta `data/` del repositorio o carga el archivo aquí."
        )
        st.stop()

    return (
        cargar_desde_upload(archivo.getvalue(), archivo.name),
        archivo.name,
    )


# ============================================================
# FILTROS Y RESÚMENES
# ============================================================

def filtrar_ambito(
    df: pd.DataFrame,
    ambito: str,
    seleccion: str | None,
) -> pd.DataFrame:
    if ambito == "Nacional" or seleccion is None:
        return df

    columna = (
        "ENTIDAD_FEDERATIVA_ANUAL"
        if ambito == "Por estado"
        else "INSTITUCION_ANUAL"
    )

    if columna not in df.columns:
        return df.iloc[0:0].copy()

    return df.loc[df[columna].eq(seleccion)].copy()


@st.cache_data(show_spinner=False)
def serie_personas_anual(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("AÑO", dropna=False)["ID_PERSONA_EXACTA"]
        .nunique()
        .reindex(range(2000, 2026), fill_value=0)
        .rename("PERSONAS")
        .reset_index()
    )



@st.cache_data(show_spinner=False)
def calcular_movimientos_anuales(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clasifica cada registro persona-año como:
    - Nuevo ingreso
    - Permanencia
    - Reingreso

    También calcula salidas observadas.
    """

    base = (
        df[
            [
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(
            [
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        )
        .copy()
    )

    base["AÑO"] = pd.to_numeric(
        base["AÑO"],
        errors="coerce",
    ).astype("Int64")

    base = base.loc[
        base["AÑO"].between(
            2000,
            2025,
            inclusive="both",
        )
    ].copy()

    base["PRIMER_AÑO_OBSERVADO"] = (
        base.groupby(
            "ID_PERSONA_EXACTA"
        )["AÑO"]
        .transform("min")
    )

    base["AÑO_ANTERIOR_OBSERVADO"] = (
        base.groupby(
            "ID_PERSONA_EXACTA"
        )["AÑO"]
        .shift()
    )

    es_nuevo = base["AÑO"].eq(
        base["PRIMER_AÑO_OBSERVADO"]
    )

    es_permanencia = (
        base["AÑO_ANTERIOR_OBSERVADO"].notna()
        & base["AÑO"].eq(
            base["AÑO_ANTERIOR_OBSERVADO"] + 1
        )
    )

    base["TIPO_MOVIMIENTO"] = np.select(
        [
            es_nuevo,
            es_permanencia,
        ],
        [
            "Nuevo ingreso",
            "Permanencia",
        ],
        default="Reingreso",
    )

    movimientos = (
        base.groupby(
            [
                "AÑO",
                "TIPO_MOVIMIENTO",
            ]
        )["ID_PERSONA_EXACTA"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            range(2000, 2026),
            fill_value=0,
        )
        .reset_index()
    )

    for columna in [
        "Nuevo ingreso",
        "Permanencia",
        "Reingreso",
    ]:
        if columna not in movimientos.columns:
            movimientos[columna] = 0

    movimientos["TOTAL_ACTIVO"] = (
        movimientos[
            [
                "Nuevo ingreso",
                "Permanencia",
                "Reingreso",
            ]
        ]
        .sum(axis=1)
        .astype(int)
    )

    personas_por_anio = {
        int(anio): set(
            grupo["ID_PERSONA_EXACTA"].astype(str)
        )
        for anio, grupo in base.groupby("AÑO")
    }

    salidas = []

    for anio in range(2000, 2026):
        if anio == 2000:
            numero_salidas = 0
        else:
            personas_previas = personas_por_anio.get(
                anio - 1,
                set(),
            )
            personas_actuales = personas_por_anio.get(
                anio,
                set(),
            )
            numero_salidas = len(
                personas_previas - personas_actuales
            )

        salidas.append(
            {
                "AÑO": anio,
                "SALIDAS_OBSERVADAS": numero_salidas,
            }
        )

    movimientos = movimientos.merge(
        pd.DataFrame(salidas),
        on="AÑO",
        how="left",
    )

    movimientos["CRECIMIENTO_NETO"] = (
        movimientos["TOTAL_ACTIVO"]
        .diff()
        .fillna(0)
        .astype(int)
    )

    detalle = base[
        [
            "ID_PERSONA_EXACTA",
            "AÑO",
            "TIPO_MOVIMIENTO",
        ]
    ].copy()

    return movimientos, detalle


@st.cache_data(show_spinner=False)
def calcular_evolucion_sexo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula personas únicas por año y sexo consolidado."""

    if "SEXO_CONSOLIDADO" not in df.columns:
        return pd.DataFrame(
            columns=[
                "AÑO",
                "SEXO",
                "PERSONAS",
            ]
        )

    base = df[
        [
            "ID_PERSONA_EXACTA",
            "AÑO",
            "SEXO_CONSOLIDADO",
        ]
    ].copy()

    base["SEXO"] = (
        base["SEXO_CONSOLIDADO"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(
            {
                "F": "MUJER",
                "FEMENINO": "MUJER",
                "M": "HOMBRE",
                "MASCULINO": "HOMBRE",
            }
        )
    )

    base["SEXO"] = base["SEXO"].where(
        base["SEXO"].isin(
            [
                "MUJER",
                "HOMBRE",
            ]
        ),
        "NO DETERMINADO",
    )

    evolucion = (
        base.dropna(
            subset=[
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        )
        .groupby(
            [
                "AÑO",
                "SEXO",
            ]
        )["ID_PERSONA_EXACTA"]
        .nunique()
        .rename("PERSONAS")
        .reset_index()
    )

    años = pd.DataFrame(
        {
            "AÑO": range(2000, 2026)
        }
    )

    sexos = pd.DataFrame(
        {
            "SEXO": [
                "MUJER",
                "HOMBRE",
                "NO DETERMINADO",
            ]
        }
    )

    estructura = años.merge(
        sexos,
        how="cross",
    )

    evolucion = estructura.merge(
        evolucion,
        on=[
            "AÑO",
            "SEXO",
        ],
        how="left",
    )

    evolucion["PERSONAS"] = (
        evolucion["PERSONAS"]
        .fillna(0)
        .astype(int)
    )

    return evolucion


def resumen_categoria(
    df: pd.DataFrame,
    columna: str,
    etiqueta: str,
) -> pd.DataFrame:
    if columna not in df.columns:
        return pd.DataFrame(columns=[etiqueta, "PERSONAS"])

    return (
        df.dropna(subset=[columna])
        .groupby(columna)["ID_PERSONA_EXACTA"]
        .nunique()
        .sort_values(ascending=False)
        .rename("PERSONAS")
        .reset_index()
        .rename(columns={columna: etiqueta})
    )


def ultimo_valor_no_nulo(
    df_persona: pd.DataFrame,
    columna: str,
) -> str:
    if columna not in df_persona.columns:
        return "Sin información"

    valores = (
        df_persona.sort_values("AÑO")[columna]
        .dropna()
        .astype("string")
        .str.strip()
    )

    return str(valores.iloc[-1]) if not valores.empty else "Sin información"


# ============================================================
# MODELOS DE PROYECCIÓN
# ============================================================

def modelo_logistico(
    x: np.ndarray,
    capacidad: float,
    tasa: float,
    punto_medio: float,
) -> np.ndarray:
    return capacidad / (1 + np.exp(-tasa * (x - punto_medio)))


def ajustar_lineal(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    modelo = LinearRegression()
    modelo.fit(x_train.reshape(-1, 1), y_train)
    return modelo.predict(x_pred.reshape(-1, 1))


def ajustar_polinomial(
    grado: int,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def _ajustar(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_pred: np.ndarray,
    ) -> np.ndarray:
        transformador = PolynomialFeatures(
            degree=grado,
            include_bias=False,
        )
        x_train_poly = transformador.fit_transform(x_train.reshape(-1, 1))
        x_pred_poly = transformador.transform(x_pred.reshape(-1, 1))

        modelo = LinearRegression()
        modelo.fit(x_train_poly, y_train)
        return modelo.predict(x_pred_poly)

    return _ajustar


def ajustar_logistico(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    capacidad_inicial = max(float(y_train.max()) * 1.25, 1.0)
    tasa_inicial = 0.15
    punto_medio_inicial = float(np.median(x_train))

    parametros, _ = curve_fit(
        modelo_logistico,
        x_train,
        y_train,
        p0=[
            capacidad_inicial,
            tasa_inicial,
            punto_medio_inicial,
        ],
        bounds=(
            [max(float(y_train.max()), 1.0), 0.0001, x_train.min() - 30],
            [max(float(y_train.max()) * 20, 10.0), 5.0, x_train.max() + 30],
        ),
        maxfev=50000,
    )

    return modelo_logistico(x_pred, *parametros)


def comparar_modelos(
    serie: pd.DataFrame,
    horizonte: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compara modelos con validación temporal y proyecta el mejor."""
    datos = serie.loc[serie["PERSONAS"].gt(0)].copy()

    if len(datos) < 8:
        raise ValueError(
            "Se requieren al menos ocho observaciones anuales con datos."
        )

    x = datos["AÑO"].astype(float).to_numpy()
    y = datos["PERSONAS"].astype(float).to_numpy()

    validacion = min(4, max(2, len(datos) // 5))
    x_train, x_test = x[:-validacion], x[-validacion:]
    y_train, y_test = y[:-validacion], y[-validacion:]

    modelos: dict[str, Callable] = {
        "Lineal": ajustar_lineal,
        "Polinomial grado 2": ajustar_polinomial(2),
        "Polinomial grado 3": ajustar_polinomial(3),
        "Logístico": ajustar_logistico,
    }

    resultados = []
    modelos_validos: dict[str, Callable] = {}

    for nombre, funcion in modelos.items():
        try:
            pred_test = np.maximum(
                funcion(x_train, y_train, x_test),
                0,
            )

            mae = mean_absolute_error(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            resultados.append(
                {
                    "MODELO": nombre,
                    "MAE": mae,
                    "RMSE": rmse,
                }
            )
            modelos_validos[nombre] = funcion

        except Exception as error:
            warnings.warn(f"{nombre} no pudo ajustarse: {error}")

    if not resultados:
        raise ValueError("Ningún modelo pudo ajustarse a la serie.")

    evaluacion = (
        pd.DataFrame(resultados)
        .sort_values(["RMSE", "MAE"])
        .reset_index(drop=True)
    )

    mejor_modelo = str(evaluacion.iloc[0]["MODELO"])
    funcion_mejor = modelos_validos[mejor_modelo]

    años_futuros = np.arange(
        int(x.max()) + 1,
        int(x.max()) + horizonte + 1,
        dtype=float,
    )

    x_completo = np.concatenate([x, años_futuros])
    pred_completa = np.maximum(
        funcion_mejor(x, y, x_completo),
        0,
    )

    proyeccion = pd.DataFrame(
        {
            "AÑO": x_completo.astype(int),
            "VALOR_MODELO": pred_completa,
            "TIPO": np.where(
                x_completo <= x.max(),
                "Ajuste histórico",
                "Proyección",
            ),
        }
    )

    observados = datos[["AÑO", "PERSONAS"]].copy()
    proyeccion = proyeccion.merge(
        observados,
        on="AÑO",
        how="left",
    )

    return evaluacion, proyeccion, mejor_modelo


# ============================================================
# MÓDULO 1: PANORAMA ACTUAL
# ============================================================

def render_panorama(df: pd.DataFrame) -> None:
    st.header("1. Panorama de la investigación en México")
    st.markdown(
        '<p class="snii-subtitle">'
        "Consulta nacional, por entidad federativa o por institución."
        "</p>",
        unsafe_allow_html=True,
    )

    controles = st.columns([1.1, 1.7, 1])

    with controles[0]:
        ambito = st.selectbox(
            "Nivel de consulta",
            ["Nacional", "Por estado", "Por institución"],
        )

    seleccion = None

    with controles[1]:
        if ambito == "Por estado":
            if "ENTIDAD_FEDERATIVA_ANUAL" not in df.columns:
                st.warning("La base no contiene entidad federativa.")
                return

            opciones = sorted(
                df["ENTIDAD_FEDERATIVA_ANUAL"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            seleccion = st.selectbox("Entidad federativa", opciones)

        elif ambito == "Por institución":
            if "INSTITUCION_ANUAL" not in df.columns:
                st.warning("La base no contiene institución.")
                return

            opciones = sorted(
                df["INSTITUCION_ANUAL"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            seleccion = st.selectbox(
                "Universidad o centro de investigación",
                opciones,
            )

        else:
            st.text_input(
                "Cobertura",
                value="Estados Unidos Mexicanos",
                disabled=True,
            )

    base_ambito = filtrar_ambito(df, ambito, seleccion)

    with controles[2]:
        años = sorted(
            base_ambito["AÑO"].dropna().astype(int).unique().tolist()
        )

        if not años:
            st.warning("No hay registros para la selección.")
            return

        año = st.selectbox(
            "Año de referencia",
            años,
            index=len(años) - 1,
        )

    actual = base_ambito.loc[base_ambito["AÑO"].eq(año)].copy()
    serie = serie_personas_anual(base_ambito)

    total_actual = actual["ID_PERSONA_EXACTA"].nunique()
    total_previo = int(
        serie.loc[serie["AÑO"].eq(año - 1), "PERSONAS"].sum()
    )
    delta = total_actual - total_previo if total_previo else None

    mujeres = (
        actual.loc[
            actual.get(
                "SEXO_CONSOLIDADO",
                pd.Series(index=actual.index, dtype="string"),
            )
            .astype("string")
            .str.upper()
            .eq("MUJER"),
            "ID_PERSONA_EXACTA",
        ].nunique()
        if "SEXO_CONSOLIDADO" in actual.columns
        else 0
    )

    porcentaje_mujeres = (
        mujeres / total_actual * 100
        if total_actual
        else 0
    )

    instituciones = (
        actual["INSTITUCION_ANUAL"].nunique(dropna=True)
        if "INSTITUCION_ANUAL" in actual.columns
        else 0
    )

    entidades = (
        actual["ENTIDAD_FEDERATIVA_ANUAL"].nunique(dropna=True)
        if "ENTIDAD_FEDERATIVA_ANUAL" in actual.columns
        else 0
    )

    metricas = st.columns(4)

    metricas[0].metric(
        f"Personas en {año}",
        f"{total_actual:,}",
        delta=f"{delta:+,}" if delta is not None else None,
    )
    metricas[1].metric(
        "Participación de mujeres",
        f"{porcentaje_mujeres:.1f}%",
        help=f"{mujeres:,} mujeres con sexo consolidado.",
    )
    metricas[2].metric("Instituciones", f"{instituciones:,}")
    metricas[3].metric("Entidades representadas", f"{entidades:,}")

    # ========================================================
    # EVOLUCIÓN HISTÓRICA Y MOVIMIENTOS
    # ========================================================

    st.subheader(
        "Evolución histórica y movimientos anuales"
    )

    movimientos, _ = calcular_movimientos_anuales(
        base_ambito
    )

    movimientos_largos = movimientos.melt(
        id_vars=[
            "AÑO",
            "SALIDAS_OBSERVADAS",
            "TOTAL_ACTIVO",
            "CRECIMIENTO_NETO",
        ],
        value_vars=[
            "Permanencia",
            "Nuevo ingreso",
            "Reingreso",
        ],
        var_name="MOVIMIENTO",
        value_name="PERSONAS",
    )

    fig_movimientos = px.bar(
        movimientos_largos,
        x="AÑO",
        y="PERSONAS",
        color="MOVIMIENTO",
        barmode="stack",
        category_orders={
            "MOVIMIENTO": [
                "Permanencia",
                "Nuevo ingreso",
                "Reingreso",
            ]
        },
        labels={
            "AÑO": "Año",
            "PERSONAS": "Personas",
            "MOVIMIENTO": "Condición anual",
        },
    )

    fig_movimientos.add_trace(
        go.Scatter(
            x=movimientos["AÑO"],
            y=movimientos["SALIDAS_OBSERVADAS"],
            mode="lines+markers",
            name="Salidas observadas",
            line=dict(
                width=3,
                dash="dot",
            ),
            marker=dict(size=7),
            hovertemplate=(
                "<b>Año %{x}</b><br>"
                "Salidas observadas: %{y:,}"
                "<extra></extra>"
            ),
        )
    )

    fig_movimientos.update_layout(
        height=520,
        hovermode="x unified",
        legend_title_text="Movimiento",
        xaxis_title="Año",
        yaxis_title="Personas",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig_movimientos,
        width="stretch",
        key="grafica_movimientos_anuales",
    )

    st.caption(
        "La barra anual suma permanencias, nuevos ingresos y "
        "reingresos. La línea muestra personas presentes en el "
        "año anterior que ya no aparecen en el año actual. "
        "La ausencia puede representar una salida real o una "
        "limitación de cobertura de la fuente."
    )

    st.subheader(
        "Evolución histórica por sexo"
    )

    evolucion_sexo = calcular_evolucion_sexo(
        base_ambito
    )

    if evolucion_sexo.empty:
        st.info(
            "No hay información de sexo disponible para "
            "construir la evolución histórica."
        )

    else:
        fig_sexo_historico = px.bar(
            evolucion_sexo,
            x="AÑO",
            y="PERSONAS",
            color="SEXO",
            barmode="stack",
            category_orders={
                "SEXO": [
                    "MUJER",
                    "HOMBRE",
                    "NO DETERMINADO",
                ]
            },
            labels={
                "AÑO": "Año",
                "PERSONAS": "Personas",
                "SEXO": "Sexo",
            },
        )

        fig_sexo_historico.update_layout(
            height=500,
            hovermode="x unified",
            legend_title_text="Sexo",
            xaxis_title="Año",
            yaxis_title="Personas",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig_sexo_historico,
            width="stretch",
            key="grafica_evolucion_sexo",
        )

        st.caption(
            "Cada barra representa a las personas únicas "
            "registradas en el año, distribuidas según el "
            "sexo consolidado disponible en la base."
        )

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader(f"Distribución por sexo · {año}")
        sexo = resumen_categoria(actual, "SEXO_CONSOLIDADO", "SEXO")

        if sexo.empty:
            st.info("No hay información de sexo para esta selección.")
        else:
            fig_sexo = px.donut(
                sexo,
                names="SEXO",
                values="PERSONAS",
                hole=0.58,
            )
            fig_sexo.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_sexo, width="stretch")

    with derecha:
        st.subheader(f"Distribución por nivel SNII · {año}")
        nivel = resumen_categoria(
            actual,
            "NIVEL_SNII_ETIQUETA",
            "NIVEL",
        )

        if nivel.empty:
            st.info("No hay información homologada de nivel.")
        else:
            fig_nivel = px.bar(
                nivel,
                x="NIVEL",
                y="PERSONAS",
                labels={"PERSONAS": "Personas"},
            )
            fig_nivel.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_nivel, width="stretch")

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader(f"Clasificación académica · {año}")
        stem = resumen_categoria(
            actual,
            "CLASIFICACION_STEM_ANUAL",
            "CLASIFICACIÓN",
        )

        if stem.empty:
            st.info("No hay clasificación académica disponible.")
        else:
            fig_stem = px.bar(
                stem.sort_values("PERSONAS"),
                x="PERSONAS",
                y="CLASIFICACIÓN",
                orientation="h",
            )
            fig_stem.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_stem, width="stretch")

    with derecha:
        st.subheader(f"Principales instituciones · {año}")
        instituciones_df = resumen_categoria(
            actual,
            "INSTITUCION_ANUAL",
            "INSTITUCIÓN",
        ).head(15)

        if instituciones_df.empty:
            st.info("No hay información institucional.")
        else:
            fig_inst = px.bar(
                instituciones_df.sort_values("PERSONAS"),
                x="PERSONAS",
                y="INSTITUCIÓN",
                orientation="h",
            )
            fig_inst.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_inst, width="stretch")

    with st.expander("Ver datos utilizados"):
        st.dataframe(
            actual,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# MÓDULO 2: PROYECCIONES
# ============================================================

def render_proyecciones(df: pd.DataFrame) -> None:
    st.header("2. Proyecciones de la población SNII")
    st.markdown(
        '<div class="snii-note">'
        "Las proyecciones son ejercicios estadísticos exploratorios. "
        "El sistema compara modelos mediante validación temporal y "
        "selecciona el de menor RMSE."
        "</div>",
        unsafe_allow_html=True,
    )

    controles = st.columns([1, 1.6, 1])

    with controles[0]:
        ambito = st.selectbox(
            "Cobertura de la proyección",
            ["Nacional", "Por estado", "Por institución"],
            key="proy_ambito",
        )

    seleccion = None

    with controles[1]:
        if ambito == "Por estado":
            opciones = sorted(
                df.get(
                    "ENTIDAD_FEDERATIVA_ANUAL",
                    pd.Series(dtype="string"),
                )
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            seleccion = st.selectbox(
                "Entidad federativa",
                opciones,
                key="proy_estado",
            )

        elif ambito == "Por institución":
            opciones = sorted(
                df.get(
                    "INSTITUCION_ANUAL",
                    pd.Series(dtype="string"),
                )
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            seleccion = st.selectbox(
                "Institución",
                opciones,
                key="proy_inst",
            )

        else:
            st.text_input(
                "Cobertura",
                value="Nacional",
                disabled=True,
                key="proy_nacional",
            )

    with controles[2]:
        horizonte = st.slider(
            "Horizonte",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )

    base_ambito = filtrar_ambito(df, ambito, seleccion)
    serie = serie_personas_anual(base_ambito)

    if serie["PERSONAS"].gt(0).sum() < 8:
        st.warning(
            "La serie no tiene suficientes años con observaciones "
            "para construir una proyección robusta."
        )
        return

    try:
        evaluacion, proyeccion, mejor_modelo = comparar_modelos(
            serie,
            horizonte,
        )
    except Exception as error:
        st.error(f"No fue posible construir la proyección: {error}")
        return

    mejor = evaluacion.iloc[0]

    metricas = st.columns(3)
    metricas[0].metric("Modelo seleccionado", mejor_modelo)
    metricas[1].metric("RMSE de validación", f"{mejor['RMSE']:,.1f}")
    metricas[2].metric("MAE de validación", f"{mejor['MAE']:,.1f}")

    fig = go.Figure()

    observados = proyeccion.loc[proyeccion["PERSONAS"].notna()]

    fig.add_trace(
        go.Scatter(
            x=observados["AÑO"],
            y=observados["PERSONAS"],
            mode="lines+markers",
            name="Observado",
        )
    )

    ajuste = proyeccion.loc[proyeccion["TIPO"].eq("Ajuste histórico")]
    futuro = proyeccion.loc[proyeccion["TIPO"].eq("Proyección")]

    fig.add_trace(
        go.Scatter(
            x=ajuste["AÑO"],
            y=ajuste["VALOR_MODELO"],
            mode="lines",
            name=f"Ajuste: {mejor_modelo}",
            line=dict(dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=futuro["AÑO"],
            y=futuro["VALOR_MODELO"],
            mode="lines+markers",
            name="Proyección",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Año",
        yaxis_title="Personas",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, width="stretch")

    izquierda, derecha = st.columns([1, 1.35])

    with izquierda:
        st.subheader("Comparación de modelos")
        tabla_evaluacion = evaluacion.copy()
        tabla_evaluacion[["MAE", "RMSE"]] = tabla_evaluacion[
            ["MAE", "RMSE"]
        ].round(2)

        st.dataframe(
            tabla_evaluacion,
            width="stretch",
            hide_index=True,
        )

    with derecha:
        st.subheader("Valores proyectados")
        tabla_futuro = futuro[["AÑO", "VALOR_MODELO"]].copy()
        tabla_futuro["VALOR_MODELO"] = (
            tabla_futuro["VALOR_MODELO"].round().astype(int)
        )
        tabla_futuro = tabla_futuro.rename(
            columns={"VALOR_MODELO": "PERSONAS_PROYECTADAS"}
        )

        st.dataframe(
            tabla_futuro,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# MÓDULO 3: INVESTIGADOR
# ============================================================

def render_investigador(df: pd.DataFrame) -> None:
    st.header("3. Historial del investigador")
    st.markdown(
        '<p class="snii-subtitle">'
        "Consulta la trayectoria anual registrada entre 2000 y 2025."
        "</p>",
        unsafe_allow_html=True,
    )

    if "NOMBRE_INVESTIGADOR" not in df.columns:
        st.warning("La base no contiene NOMBRE_INVESTIGADOR.")
        return

    catalogo = (
        df[
            [
                "ID_PERSONA_EXACTA",
                "NOMBRE_INVESTIGADOR",
                *(
                    ["CVU_REFERENCIA"]
                    if "CVU_REFERENCIA" in df.columns
                    else []
                ),
            ]
        ]
        .drop_duplicates("ID_PERSONA_EXACTA")
        .dropna(subset=["NOMBRE_INVESTIGADOR"])
        .sort_values("NOMBRE_INVESTIGADOR")
        .copy()
    )

    catalogo["ETIQUETA"] = (
        catalogo["NOMBRE_INVESTIGADOR"].astype(str)
        + " · "
        + catalogo["ID_PERSONA_EXACTA"].astype(str)
    )

    busqueda = st.text_input(
        "Buscar por nombre, CVU o ID",
        placeholder="Escribe al menos tres caracteres…",
    ).strip()

    if len(busqueda) < 3:
        st.info("Escribe al menos tres caracteres para buscar.")
        return

    mascara = (
        catalogo["ETIQUETA"]
        .astype(str)
        .str.contains(busqueda, case=False, na=False, regex=False)
    )

    if "CVU_REFERENCIA" in catalogo.columns:
        mascara = mascara | (
            catalogo["CVU_REFERENCIA"]
            .astype(str)
            .str.contains(busqueda, case=False, na=False, regex=False)
        )

    coincidencias = catalogo.loc[mascara].head(100)

    if coincidencias.empty:
        st.warning("No se encontraron coincidencias.")
        return

    etiqueta = st.selectbox(
        "Selecciona una persona",
        coincidencias["ETIQUETA"].tolist(),
    )

    id_persona = coincidencias.loc[
        coincidencias["ETIQUETA"].eq(etiqueta),
        "ID_PERSONA_EXACTA",
    ].iloc[0]

    historial = (
        df.loc[df["ID_PERSONA_EXACTA"].eq(id_persona)]
        .sort_values("AÑO")
        .copy()
    )

    nombre = ultimo_valor_no_nulo(
        historial,
        "NOMBRE_INVESTIGADOR",
    )
    sexo = ultimo_valor_no_nulo(historial, "SEXO_CONSOLIDADO")
    institucion = ultimo_valor_no_nulo(historial, "INSTITUCION_ANUAL")
    entidad = ultimo_valor_no_nulo(
        historial,
        "ENTIDAD_FEDERATIVA_ANUAL",
    )

    primer_año = int(historial["AÑO"].min())
    ultimo_año = int(historial["AÑO"].max())
    años_presentes = historial["AÑO"].nunique()

    st.subheader(nombre)

    metricas = st.columns(4)
    metricas[0].metric("Primer año observado", primer_año)
    metricas[1].metric("Último año observado", ultimo_año)
    metricas[2].metric("Años con registro", años_presentes)
    metricas[3].metric("Sexo consolidado", sexo)

    st.markdown(
        f"""
        **Institución más reciente:** {institucion}  
        **Entidad más reciente:** {entidad}
        """
    )

    if "NIVEL_SNII_STD" in historial.columns:
        historial_nivel = historial.dropna(subset=["NIVEL_SNII_STD"]).copy()

        if not historial_nivel.empty:
            nivel_maximo = int(historial_nivel["NIVEL_SNII_STD"].max())
            nivel_ultimo = ultimo_valor_no_nulo(
                historial_nivel,
                "NIVEL_SNII_ETIQUETA",
            )

            cols = st.columns(3)
            cols[0].metric("Nivel más reciente", nivel_ultimo)
            cols[1].metric("Código máximo histórico", nivel_maximo)

            años_posibles = ultimo_año - primer_año + 1
            continuidad = (
                años_presentes / años_posibles * 100
                if años_posibles > 0
                else 0
            )
            cols[2].metric("Continuidad observada", f"{continuidad:.1f}%")

    st.subheader("Trayectoria de nivel")

    if "NIVEL_SNII_STD" in historial.columns:
        datos_nivel = historial.dropna(subset=["NIVEL_SNII_STD"])

        if datos_nivel.empty:
            st.info("No existe nivel homologado para esta persona.")
        else:
            fig_nivel = px.line(
                datos_nivel,
                x="AÑO",
                y="NIVEL_SNII_STD",
                markers=True,
                hover_data=[
                    col
                    for col in [
                        "NIVEL_SNII_ETIQUETA",
                        "INSTITUCION_ANUAL",
                        "ENTIDAD_FEDERATIVA_ANUAL",
                    ]
                    if col in datos_nivel.columns
                ],
            )
            fig_nivel.update_yaxes(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=[
                    "Candidato",
                    "Nivel I",
                    "Nivel II",
                    "Nivel III",
                    "Emérito",
                ],
            )
            fig_nivel.update_layout(
                height=430,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_nivel, width="stretch")

    st.subheader("Estimación exploratoria de continuidad")

    años_posibles = ultimo_año - primer_año + 1
    continuidad_historica = (
        años_presentes / años_posibles
        if años_posibles > 0
        else 0
    )

    últimos_cinco = set(range(max(primer_año, ultimo_año - 4), ultimo_año + 1))
    observados_ultimos = set(historial["AÑO"].astype(int).tolist())
    continuidad_reciente = (
        len(últimos_cinco & observados_ultimos)
        / len(últimos_cinco)
        if últimos_cinco
        else 0
    )

    probabilidad_exploratoria = (
        0.4 * continuidad_historica
        + 0.6 * continuidad_reciente
    ) * 100

    st.metric(
        "Índice exploratorio de presencia futura",
        f"{probabilidad_exploratoria:.1f}%",
        help=(
            "No es todavía un modelo predictivo entrenado. Combina "
            "continuidad histórica y presencia reciente como referencia."
        ),
    )

    st.caption(
        "La predicción individual formal se incorporará después de "
        "construir variables de cohorte, promociones, interrupciones "
        "y permanencia por nivel."
    )

    columnas_historial = [
        col
        for col in [
            "AÑO",
            "NIVEL_SNII_ETIQUETA",
            "INSTITUCION_ANUAL",
            "DEPENDENCIA_ANUAL",
            "ENTIDAD_FEDERATIVA_ANUAL",
            "AREA_DEL_CONOCIMIENTO_ANUAL",
            "DISCIPLINA_ANUAL",
            "CLASIFICACION_STEM_ANUAL",
        ]
        if col in historial.columns
    ]

    with st.expander("Ver historial anual"):
        st.dataframe(
            historial[columnas_historial],
            width="stretch",
            hide_index=True,
        )



# ============================================================
# MÓDULO 4: LABORATORIO DE VISUALIZACIÓN
# ============================================================

# El catálogo conserva las 69 variables procesadas en el master.
# Las variables de identificación pueden utilizarse como filtros,
# pero no se recomiendan como ejes gráficos cuando tienen alta cardinalidad.

CATALOGO_VARIABLES = {
    # --------------------------------------------------------
    # IDENTIDAD
    # --------------------------------------------------------
    "Identificador exacto de persona": {
        "columna": "ID_PERSONA_EXACTA",
        "tipo": "identificador",
        "familia": "Identidad",
    },
    "CVU de referencia": {
        "columna": "CVU_REFERENCIA",
        "tipo": "identificador",
        "familia": "Identidad",
    },
    "Nombre de la persona investigadora": {
        "columna": "NOMBRE_INVESTIGADOR",
        "tipo": "identificador",
        "familia": "Identidad",
    },
    "Nombre completo de referencia": {
        "columna": "NOMBRE_COMPLETO_REFERENCIA",
        "tipo": "identificador",
        "familia": "Identidad",
    },
    "Apellido paterno": {
        "columna": "APELLIDO_PATERNO_REFERENCIA_EXPEDIENTE",
        "tipo": "categorica",
        "familia": "Identidad",
    },
    "Apellido materno": {
        "columna": "APELLIDO_MATERNO_REFERENCIA_EXPEDIENTE",
        "tipo": "categorica",
        "familia": "Identidad",
    },
    "Nombres": {
        "columna": "NOMBRES_REFERENCIA_EXPEDIENTE",
        "tipo": "categorica",
        "familia": "Identidad",
    },
    "Nobilis consolidado": {
        "columna": "NOBILIS_CONSOLIDADO",
        "tipo": "categorica",
        "familia": "Identidad",
    },
    "Estado de identidad consolidada": {
        "columna": "ESTADO_IDENTIDAD_CONSOLIDADA",
        "tipo": "categorica",
        "familia": "Identidad",
    },
    "Requiere revisión de identidad": {
        "columna": "REQUIERE_REVISION_IDENTIDAD_CONSOLIDADA",
        "tipo": "binaria",
        "familia": "Identidad",
    },

    # --------------------------------------------------------
    # TIEMPO Y TRAYECTORIA
    # --------------------------------------------------------
    "Año": {
        "columna": "AÑO",
        "tipo": "temporal",
        "familia": "Trayectoria",
    },
    "Primer año observado": {
        "columna": "PRIMER_AÑO",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Último año observado": {
        "columna": "ULTIMO_AÑO",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Número de años presente": {
        "columna": "NUMERO_AÑOS_PRESENTE",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Años desde el primer registro": {
        "columna": "AÑO_DESDE_PRIMER_REGISTRO",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Años hasta el último registro": {
        "columna": "AÑO_HASTA_ULTIMO_REGISTRO",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Antigüedad acumulada": {
        "columna": "ANTIGUEDAD_ACUMULADA_AÑOS",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Es primer año de la persona": {
        "columna": "ES_PRIMER_AÑO_PERSONA",
        "tipo": "binaria",
        "familia": "Trayectoria",
    },
    "Es último año de la persona": {
        "columna": "ES_ULTIMO_AÑO_PERSONA",
        "tipo": "binaria",
        "familia": "Trayectoria",
    },
    "Vigente en 2025": {
        "columna": "ESTA_VIGENTE_EN_2025",
        "tipo": "binaria",
        "familia": "Trayectoria",
    },
    "Número de registros anuales": {
        "columna": "NUMERO_REGISTROS_ANUALES",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Número de registros": {
        "columna": "NUMERO_REGISTROS",
        "tipo": "numerica",
        "familia": "Trayectoria",
    },
    "Categoría anual": {
        "columna": "CATEGORIA_ANUAL",
        "tipo": "categorica",
        "familia": "Trayectoria",
    },

    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------
    "Sexo consolidado": {
        "columna": "SEXO_CONSOLIDADO",
        "tipo": "categorica",
        "familia": "Sexo",
    },
    "Fuente del sexo": {
        "columna": "FUENTE_SEXO",
        "tipo": "categorica",
        "familia": "Sexo",
    },
    "Confianza del sexo": {
        "columna": "CONFIANZA_SEXO",
        "tipo": "categorica",
        "familia": "Sexo",
    },
    "Requiere revisión de sexo": {
        "columna": "REQUIERE_REVISION_SEXO",
        "tipo": "binaria",
        "familia": "Sexo",
    },

    # --------------------------------------------------------
    # UBICACIÓN E INSTITUCIÓN
    # --------------------------------------------------------
    "Institución anual": {
        "columna": "INSTITUCION_ANUAL",
        "tipo": "categorica",
        "familia": "Ubicación e institución",
    },
    "Dependencia anual": {
        "columna": "DEPENDENCIA_ANUAL",
        "tipo": "categorica",
        "familia": "Ubicación e institución",
    },
    "Subdependencia anual": {
        "columna": "SUBDEPENDENCIA_ANUAL",
        "tipo": "categorica",
        "familia": "Ubicación e institución",
    },
    "Entidad federativa anual": {
        "columna": "ENTIDAD_FEDERATIVA_ANUAL",
        "tipo": "categorica",
        "familia": "Ubicación e institución",
    },
    "País anual": {
        "columna": "PAIS_ANUAL",
        "tipo": "categorica",
        "familia": "Ubicación e institución",
    },

    # --------------------------------------------------------
    # NIVEL SNII
    # --------------------------------------------------------
    "Nivel anual original": {
        "columna": "NIVEL_ANUAL",
        "tipo": "categorica",
        "familia": "Nivel SNII",
    },
    "Código homologado de nivel SNII": {
        "columna": "NIVEL_SNII_STD",
        "tipo": "ordinal",
        "familia": "Nivel SNII",
    },
    "Nivel SNII homologado": {
        "columna": "NIVEL_SNII_ETIQUETA",
        "tipo": "categorica",
        "familia": "Nivel SNII",
    },
    "Estado de homologación del nivel": {
        "columna": "ESTADO_HOMOLOGACION_NIVEL",
        "tipo": "categorica",
        "familia": "Nivel SNII",
    },
    "Texto comparable de nivel": {
        "columna": "_NIVEL_TEXTO_COMPARABLE",
        "tipo": "categorica",
        "familia": "Nivel SNII",
    },
    "Regla de homologación del nivel": {
        "columna": "REGLA_HOMOLOGACION_NIVEL",
        "tipo": "categorica",
        "familia": "Nivel SNII",
    },

    # --------------------------------------------------------
    # CLASIFICACIÓN ACADÉMICA
    # --------------------------------------------------------
    "Área del conocimiento": {
        "columna": "AREA_DEL_CONOCIMIENTO_ANUAL",
        "tipo": "categorica",
        "familia": "Clasificación académica",
    },
    "Campo del conocimiento": {
        "columna": "CAMPO_DEL_CONOCIMIENTO_ANUAL",
        "tipo": "categorica",
        "familia": "Clasificación académica",
    },
    "Disciplina": {
        "columna": "DISCIPLINA_ANUAL",
        "tipo": "categorica",
        "familia": "Clasificación académica",
    },
    "Subdisciplina": {
        "columna": "SUBDISCIPLINA_ANUAL",
        "tipo": "categorica",
        "familia": "Clasificación académica",
    },
    "Especialidad": {
        "columna": "ESPECIALIDAD_ANUAL",
        "tipo": "categorica",
        "familia": "Clasificación académica",
    },

    # --------------------------------------------------------
    # STEM Y GRANDES GRUPOS
    # --------------------------------------------------------
    "Clasificación académica STEM ampliada": {
        "columna": "CLASIFICACION_STEM_ANUAL",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "STEM frente a no STEM": {
        "columna": "GRUPO_STEM_BINARIO",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Es STEM estricto": {
        "columna": "ES_STEM_ESTRICTO",
        "tipo": "binaria",
        "familia": "STEM",
    },
    "Es STEM ampliado": {
        "columna": "ES_STEM_AMPLIADO",
        "tipo": "binaria",
        "familia": "STEM",
    },
    "Confianza de clasificación STEM": {
        "columna": "CONFIANZA_CLASIFICACION_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Palabras coincidentes STEM": {
        "columna": "PALABRAS_COINCIDENTES_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Requiere revisión STEM": {
        "columna": "REQUIERE_REVISION_STEM",
        "tipo": "binaria",
        "familia": "STEM",
    },
    "Área utilizada para STEM": {
        "columna": "_AREA_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Campo utilizado para STEM": {
        "columna": "_CAMPO_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Disciplina utilizada para STEM": {
        "columna": "_DISCIPLINA_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Subdisciplina utilizada para STEM": {
        "columna": "_SUBDISCIPLINA_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Especialidad utilizada para STEM": {
        "columna": "_ESPECIALIDAD_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Número de fuentes académicas STEM": {
        "columna": "NUMERO_FUENTES_ACADEMICAS_STEM",
        "tipo": "numerica",
        "familia": "STEM",
    },
    "Texto académico STEM": {
        "columna": "TEXTO_ACADEMICO_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Clave de texto académico STEM": {
        "columna": "TEXTO_ACADEMICO_STEM_CLAVE",
        "tipo": "categorica",
        "familia": "STEM",
    },
    "Puntaje STEM": {
        "columna": "PUNTAJE_STEM",
        "tipo": "numerica",
        "familia": "STEM",
    },
    "Puntaje salud y biológicas": {
        "columna": "PUNTAJE_SALUD_BIOLOGICAS",
        "tipo": "numerica",
        "familia": "STEM",
    },
    "Puntaje ciencias sociales": {
        "columna": "PUNTAJE_SOCIALES",
        "tipo": "numerica",
        "familia": "STEM",
    },
    "Puntaje humanidades y artes": {
        "columna": "PUNTAJE_HUMANIDADES_ARTES",
        "tipo": "numerica",
        "familia": "STEM",
    },
    "Fuente de clasificación STEM": {
        "columna": "FUENTE_CLASIFICACION_STEM",
        "tipo": "categorica",
        "familia": "STEM",
    },

    # --------------------------------------------------------
    # CALIDAD Y COMPLETITUD
    # --------------------------------------------------------
    "Campos clave disponibles": {
        "columna": "NUMERO_CAMPOS_CLAVE_DISPONIBLES",
        "tipo": "numerica",
        "familia": "Calidad de datos",
    },
    "Campos clave faltantes": {
        "columna": "NUMERO_CAMPOS_CLAVE_FALTANTES",
        "tipo": "numerica",
        "familia": "Calidad de datos",
    },
    "Porcentaje de completitud": {
        "columna": "PORCENTAJE_COMPLETITUD_CLAVE",
        "tipo": "numerica",
        "familia": "Calidad de datos",
    },
    "Requiere revisión del master": {
        "columna": "REQUIERE_REVISION_MASTER",
        "tipo": "binaria",
        "familia": "Calidad de datos",
    },
    "Estado del registro master": {
        "columna": "ESTADO_REGISTRO_MASTER",
        "tipo": "categorica",
        "familia": "Calidad de datos",
    },
    "Requiere revisión del expediente": {
        "columna": "REQUIERE_REVISION_EXPEDIENTE",
        "tipo": "binaria",
        "familia": "Calidad de datos",
    },
}


def normalizar_binaria_laboratorio(
    serie: pd.Series,
) -> pd.Series:
    """Homologa valores booleanos o binarios para visualización."""

    texto = (
        serie.astype("string")
        .str.strip()
        .str.upper()
    )

    return texto.replace(
        {
            "TRUE": "Sí",
            "FALSE": "No",
            "1": "Sí",
            "0": "No",
            "SI": "Sí",
            "SÍ": "Sí",
            "NO": "No",
        }
    )


def preparar_base_laboratorio(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepara variables numéricas, binarias y temporales."""

    base = df.copy()

    for nombre, especificacion in CATALOGO_VARIABLES.items():
        columna = especificacion["columna"]
        tipo = especificacion["tipo"]

        if columna not in base.columns:
            continue

        if tipo in {
            "numerica",
            "ordinal",
            "temporal",
        }:
            base[columna] = pd.to_numeric(
                base[columna],
                errors="coerce",
            )

        elif tipo == "binaria":
            base[columna] = normalizar_binaria_laboratorio(
                base[columna]
            )

        else:
            base[columna] = (
                base[columna]
                .astype("string")
                .str.strip()
            )

    return base


def construir_catalogo_disponible(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un inventario de variables:
    - disponibles con datos,
    - existentes pero vacías,
    - ausentes del archivo.
    """

    filas = []

    for nombre, especificacion in CATALOGO_VARIABLES.items():
        columna = especificacion["columna"]

        existe = columna in df.columns
        con_datos = (
            bool(df[columna].notna().any())
            if existe
            else False
        )

        filas.append(
            {
                "VARIABLE": nombre,
                "COLUMNA": columna,
                "FAMILIA": especificacion["familia"],
                "TIPO": especificacion["tipo"],
                "EXISTE": existe,
                "CON_DATOS": con_datos,
            }
        )

    return pd.DataFrame(filas)


def opciones_variables_por_familia(
    inventario: pd.DataFrame,
    incluir_identificadores: bool = False,
) -> list[str]:
    """Devuelve opciones agrupables y realmente disponibles."""

    disponibles = inventario.loc[
        inventario["CON_DATOS"]
    ].copy()

    if not incluir_identificadores:
        disponibles = disponibles.loc[
            disponibles["TIPO"].ne(
                "identificador"
            )
        ]

    return (
        disponibles.sort_values(
            [
                "FAMILIA",
                "VARIABLE",
            ]
        )["VARIABLE"]
        .tolist()
    )


def filtrar_escala_laboratorio(
    df: pd.DataFrame,
    escala: str,
    seleccion: str | None,
) -> pd.DataFrame:
    """Aplica escala nacional, estatal o institucional."""

    if escala == "Nacional":
        return df

    if escala == "Por estado":
        columna = "ENTIDAD_FEDERATIVA_ANUAL"
    else:
        columna = "INSTITUCION_ANUAL"

    if (
        columna not in df.columns
        or not df[columna].notna().any()
        or seleccion is None
    ):
        return df.iloc[0:0].copy()

    return df.loc[
        df[columna].eq(seleccion)
    ].copy()


def recomendar_visualizaciones(
    variables: list[str],
    usa_tiempo: bool,
) -> list[dict[str, object]]:
    """Recomienda gráficas con reglas transparentes."""

    tipos = [
        CATALOGO_VARIABLES[variable]["tipo"]
        for variable in variables
    ]

    tipos_analiticos = [
        "categorica"
        if tipo in {
            "categorica",
            "binaria",
            "ordinal",
        }
        else tipo
        for tipo in tipos
    ]

    n_cat = tipos_analiticos.count(
        "categorica"
    )

    n_num = tipos_analiticos.count(
        "numerica"
    )

    recomendaciones = []

    def agregar(
        nombre: str,
        score: int,
        razon: str,
    ) -> None:
        recomendaciones.append(
            {
                "grafica": nombre,
                "idoneidad": score,
                "razon": razon,
            }
        )

    if usa_tiempo:
        if n_cat == 1 and n_num == 0:
            agregar(
                "Barras apiladas",
                97,
                "Muestra la composición anual de las categorías.",
            )
            agregar(
                "Líneas múltiples",
                89,
                "Permite seguir la tendencia de cada categoría.",
            )
            agregar(
                "Área apilada",
                82,
                "Destaca el crecimiento total y la composición.",
            )

        elif n_cat >= 2 and n_num == 0:
            agregar(
                "Mapa de calor",
                94,
                "Resume dos categorías a través del tiempo sin saturar barras.",
            )
            agregar(
                "Barras apiladas",
                88,
                "Funciona cuando se limitan las categorías principales.",
            )

        elif n_num >= 1 and n_cat >= 1:
            agregar(
                "Boxplot por periodo",
                95,
                "Compara una distribución numérica por grupo y año.",
            )
            agregar(
                "Líneas de mediana",
                89,
                "Resume la evolución de la tendencia central.",
            )

        elif n_num >= 1:
            agregar(
                "Línea temporal",
                96,
                "Representa la evolución anual de una variable numérica.",
            )
            agregar(
                "Área",
                78,
                "Enfatiza la magnitud de la tendencia.",
            )

    else:
        if n_cat == 1 and n_num == 0:
            agregar(
                "Dona",
                96,
                "Muestra la composición de una variable con pocas categorías.",
            )
            agregar(
                "Barras",
                94,
                "Compara con precisión el tamaño de las categorías.",
            )

        elif n_cat >= 2 and n_num == 0:
            agregar(
                "Barras apiladas",
                95,
                "Compara la composición conjunta de dos variables categóricas.",
            )
            agregar(
                "Mapa de calor",
                91,
                "Hace visibles concentraciones entre categorías.",
            )

        elif n_num == 1 and n_cat == 0:
            agregar(
                "Histograma",
                98,
                "Representa la distribución de una variable numérica.",
            )
            agregar(
                "Boxplot",
                91,
                "Resume mediana, dispersión y valores atípicos.",
            )

        elif n_num >= 1 and n_cat >= 1:
            agregar(
                "Boxplot",
                97,
                "Compara una variable numérica entre categorías.",
            )
            agregar(
                "Violin plot",
                89,
                "Muestra la forma de la distribución por grupo.",
            )
            agregar(
                "Burbujas",
                74,
                "Resume categorías, valores y tamaño en tres dimensiones.",
            )

        elif n_num >= 2:
            agregar(
                "Dispersión",
                97,
                "Muestra la relación entre dos variables numéricas.",
            )
            agregar(
                "Burbujas",
                87,
                "Permite incorporar una tercera dimensión.",
            )

    return sorted(
        recomendaciones,
        key=lambda fila: fila["idoneidad"],
        reverse=True,
    )


def limitar_categorias(
    df: pd.DataFrame,
    columna: str,
    maximo: int = 12,
) -> pd.DataFrame:
    """Conserva categorías principales y agrupa el resto como Otros."""

    if columna not in df.columns:
        return df

    conteos = (
        df.groupby(columna)["ID_PERSONA_EXACTA"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    if len(conteos) <= maximo:
        return df

    principales = set(
        conteos.head(maximo).index
    )

    salida = df.copy()

    salida[columna] = salida[columna].where(
        salida[columna].isin(
            principales
        ),
        "OTROS",
    )

    return salida


def construir_dataset_laboratorio(
    df: pd.DataFrame,
    variables: list[str],
    usa_tiempo: bool,
    periodo: tuple[int, int] | None,
    anio: int | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepara el conjunto de análisis."""

    columnas = [
        CATALOGO_VARIABLES[variable]["columna"]
        for variable in variables
    ]

    tipos = [
        CATALOGO_VARIABLES[variable]["tipo"]
        for variable in variables
    ]

    base = df.copy()

    if usa_tiempo and periodo is not None:
        base = base.loc[
            base["AÑO"].between(
                periodo[0],
                periodo[1],
                inclusive="both",
            )
        ].copy()

    elif not usa_tiempo and anio is not None:
        base = base.loc[
            base["AÑO"].eq(
                anio
            )
        ].copy()

    columnas_presentes = [
        columna
        for columna in columnas
        if columna in base.columns
    ]

    base = base[
        list(
            dict.fromkeys(
                [
                    "ID_PERSONA_EXACTA",
                    "AÑO",
                    *columnas_presentes,
                ]
            )
        )
    ].copy()

    base = base.dropna(
        subset=columnas_presentes,
        how="any",
    )

    for columna, tipo in zip(
        columnas_presentes,
        tipos,
    ):
        if tipo in {
            "categorica",
            "binaria",
            "ordinal",
        }:
            base = limitar_categorias(
                base,
                columna,
                maximo=12,
            )

    return (
        base,
        columnas_presentes,
        tipos,
    )


def generar_figura_laboratorio(
    base: pd.DataFrame,
    variables: list[str],
    columnas: list[str],
    tipos: list[str],
    usa_tiempo: bool,
    grafica: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Genera la gráfica seleccionada."""

    categoricas = [
        (
            nombre,
            columna,
        )
        for nombre, columna, tipo
        in zip(
            variables,
            columnas,
            tipos,
        )
        if tipo in {
            "categorica",
            "binaria",
            "ordinal",
        }
    ]

    numericas = [
        (
            nombre,
            columna,
        )
        for nombre, columna, tipo
        in zip(
            variables,
            columnas,
            tipos,
        )
        if tipo == "numerica"
    ]

    if grafica == "Dona":
        nombre, columna = categoricas[0]

        resumen = (
            base.groupby(columna)["ID_PERSONA_EXACTA"]
            .nunique()
            .rename("PERSONAS")
            .reset_index()
        )

        figura = px.pie(
            resumen,
            names=columna,
            values="PERSONAS",
            hole=0.58,
            labels={
                columna: nombre,
                "PERSONAS": "Personas",
            },
        )

        figura.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        return figura, resumen

    if grafica == "Barras":
        nombre, columna = categoricas[0]

        resumen = (
            base.groupby(columna)["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_values(
                ascending=False
            )
            .rename("PERSONAS")
            .reset_index()
        )

        figura = px.bar(
            resumen,
            x=columna,
            y="PERSONAS",
            labels={
                columna: nombre,
                "PERSONAS": "Personas",
            },
        )

        return figura, resumen

    if grafica in {
        "Barras apiladas",
        "Líneas múltiples",
        "Área apilada",
    }:
        nombre_color, columna_color = categoricas[0]

        resumen = (
            base.groupby(
                [
                    "AÑO",
                    columna_color,
                ]
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .rename("PERSONAS")
            .reset_index()
        )

        if grafica == "Barras apiladas":
            figura = px.bar(
                resumen,
                x="AÑO",
                y="PERSONAS",
                color=columna_color,
                barmode="stack",
                labels={
                    columna_color: nombre_color,
                    "PERSONAS": "Personas",
                },
            )

        elif grafica == "Líneas múltiples":
            figura = px.line(
                resumen,
                x="AÑO",
                y="PERSONAS",
                color=columna_color,
                markers=True,
                labels={
                    columna_color: nombre_color,
                    "PERSONAS": "Personas",
                },
            )

        else:
            figura = px.area(
                resumen,
                x="AÑO",
                y="PERSONAS",
                color=columna_color,
                labels={
                    columna_color: nombre_color,
                    "PERSONAS": "Personas",
                },
            )

        return figura, resumen

    if grafica == "Mapa de calor":
        if usa_tiempo:
            nombre, columna = categoricas[0]

            resumen = (
                base.groupby(
                    [
                        columna,
                        "AÑO",
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            matriz = resumen.pivot(
                index=columna,
                columns="AÑO",
                values="PERSONAS",
            ).fillna(0)

        else:
            _, columna_1 = categoricas[0]
            _, columna_2 = categoricas[1]

            resumen = (
                base.groupby(
                    [
                        columna_1,
                        columna_2,
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            matriz = resumen.pivot(
                index=columna_1,
                columns=columna_2,
                values="PERSONAS",
            ).fillna(0)

        figura = px.imshow(
            matriz,
            aspect="auto",
            labels={
                "color": "Personas",
            },
        )

        return figura, resumen

    if grafica == "Histograma":
        nombre, columna = numericas[0]

        figura = px.histogram(
            base,
            x=columna,
            nbins=30,
            labels={
                columna: nombre,
                "count": "Registros",
            },
        )

        return (
            figura,
            base[
                [
                    columna,
                ]
            ].copy(),
        )

    if grafica in {
        "Boxplot",
        "Boxplot por periodo",
        "Violin plot",
    }:
        nombre_num, columna_num = numericas[0]

        if categoricas:
            nombre_cat, columna_cat = categoricas[0]
        else:
            nombre_cat = "Año"
            columna_cat = "AÑO"

        color = (
            categoricas[1][1]
            if len(categoricas) >= 2
            else None
        )

        if grafica == "Violin plot":
            figura = px.violin(
                base,
                x=columna_cat,
                y=columna_num,
                color=color,
                box=True,
                points=False,
                labels={
                    columna_cat: nombre_cat,
                    columna_num: nombre_num,
                },
            )

        else:
            figura = px.box(
                base,
                x=columna_cat,
                y=columna_num,
                color=color,
                points="outliers",
                labels={
                    columna_cat: nombre_cat,
                    columna_num: nombre_num,
                },
            )

        columnas_salida = [
            columna_cat,
            columna_num,
        ]

        if color is not None:
            columnas_salida.append(
                color
            )

        return (
            figura,
            base[
                columnas_salida
            ].copy(),
        )

    if grafica in {
        "Línea temporal",
        "Líneas de mediana",
        "Área",
    }:
        nombre_num, columna_num = numericas[0]

        agrupadores = [
            "AÑO",
        ]

        color = None

        if categoricas:
            color = categoricas[0][1]
            agrupadores.append(
                color
            )

        resumen = (
            base.groupby(
                agrupadores
            )[columna_num]
            .median()
            .rename("MEDIANA")
            .reset_index()
        )

        if grafica == "Área":
            figura = px.area(
                resumen,
                x="AÑO",
                y="MEDIANA",
                color=color,
                labels={
                    "MEDIANA":
                        f"Mediana de {nombre_num}",
                },
            )
        else:
            figura = px.line(
                resumen,
                x="AÑO",
                y="MEDIANA",
                color=color,
                markers=True,
                labels={
                    "MEDIANA":
                        f"Mediana de {nombre_num}",
                },
            )

        return figura, resumen

    if grafica in {
        "Dispersión",
        "Burbujas",
    }:
        nombre_x, columna_x = numericas[0]
        nombre_y, columna_y = numericas[1]

        color = (
            categoricas[0][1]
            if categoricas
            else None
        )

        columnas_salida = [
            columna_x,
            columna_y,
        ]

        if color is not None:
            columnas_salida.append(
                color
            )

        resumen = base[
            columnas_salida
        ].copy()

        figura = px.scatter(
            resumen,
            x=columna_x,
            y=columna_y,
            color=color,
            size=(
                columna_y
                if grafica == "Burbujas"
                else None
            ),
            labels={
                columna_x: nombre_x,
                columna_y: nombre_y,
            },
            size_max=35,
        )

        return figura, resumen

    raise ValueError(
        f"La gráfica '{grafica}' todavía no está implementada."
    )


def interpretar_laboratorio(
    base: pd.DataFrame,
    variables: list[str],
    columnas: list[str],
    tipos: list[str],
    escala: str,
) -> tuple[str, str]:
    """Genera descripción e interpretación breve."""

    descripcion = (
        f"La gráfica resume los registros del SNII a escala "
        f"{escala.lower()}, utilizando personas únicas como "
        "unidad de conteo cuando la selección es categórica."
    )

    categoricas = [
        (
            nombre,
            columna,
        )
        for nombre, columna, tipo
        in zip(
            variables,
            columnas,
            tipos,
        )
        if tipo in {
            "categorica",
            "binaria",
            "ordinal",
        }
    ]

    numericas = [
        (
            nombre,
            columna,
        )
        for nombre, columna, tipo
        in zip(
            variables,
            columnas,
            tipos,
        )
        if tipo == "numerica"
    ]

    if categoricas:
        nombre, columna = categoricas[0]

        resumen = (
            base.groupby(columna)["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        if not resumen.empty:
            principal = resumen.index[0]
            cantidad = int(
                resumen.iloc[0]
            )
            total = int(
                resumen.sum()
            )
            porcentaje = (
                cantidad / total * 100
                if total
                else 0
            )

            interpretacion = (
                f"En {nombre.lower()}, la categoría con mayor "
                f"representación es '{principal}', con "
                f"{cantidad:,} personas ({porcentaje:.1f}% del "
                "total clasificado en la selección)."
            )

            return (
                descripcion,
                interpretacion,
            )

    if numericas:
        nombre, columna = numericas[0]

        valores = base[
            columna
        ].dropna()

        if not valores.empty:
            interpretacion = (
                f"La mediana de {nombre.lower()} es "
                f"{valores.median():.1f}. El 50% central de "
                f"los registros se encuentra entre "
                f"{valores.quantile(0.25):.1f} y "
                f"{valores.quantile(0.75):.1f}."
            )

            return (
                descripcion,
                interpretacion,
            )

    return (
        descripcion,
        "La selección permite identificar patrones descriptivos. "
        "Los resultados deben interpretarse considerando la "
        "cobertura y completitud de la fuente.",
    )


def render_laboratorio_visualizacion(
    df: pd.DataFrame,
) -> None:
    """Constructor guiado de análisis y visualizaciones."""

    st.header(
        "4. Laboratorio de visualización"
    )

    st.markdown(
        '<p class="snii-subtitle">'
        "Construye una pregunta con las variables procesadas "
        "en el master y recibe una recomendación gráfica."
        "</p>",
        unsafe_allow_html=True,
    )

    base = preparar_base_laboratorio(
        df
    )

    inventario = construir_catalogo_disponible(
        base
    )

    variables_disponibles = (
        opciones_variables_por_familia(
            inventario,
            incluir_identificadores=False,
        )
    )

    st.markdown(
        """
        <span class="lab-chip lab-objetivo">Objetivo</span>
        <span class="lab-chip lab-variable">Variable</span>
        <span class="lab-chip lab-tiempo">Tiempo</span>
        <span class="lab-chip lab-filtro">Filtro</span>
        <span class="lab-chip lab-ubicacion">Escala geográfica</span>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "Inventario de variables procesadas"
    ):
        inventario_mostrar = inventario.copy()

        inventario_mostrar[
            "ESTADO"
        ] = np.select(
            [
                inventario_mostrar["CON_DATOS"],
                inventario_mostrar["EXISTE"],
            ],
            [
                "Disponible",
                "Existe, pero está vacía",
            ],
            default="No está en el archivo",
        )

        st.dataframe(
            inventario_mostrar[
                [
                    "FAMILIA",
                    "VARIABLE",
                    "COLUMNA",
                    "TIPO",
                    "ESTADO",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # ESCALA GEOGRÁFICA
    # --------------------------------------------------------

    st.subheader(
        "Construye tu propuesta"
    )

    escala_columnas = st.columns(
        [
            1,
            2,
        ]
    )

    with escala_columnas[0]:
        st.markdown(
            '<span class="lab-chip lab-ubicacion">Escala</span>',
            unsafe_allow_html=True,
        )

        escala = st.selectbox(
            "Escala del análisis",
            [
                "Nacional",
                "Por estado",
                "Por institución o centro",
            ],
            label_visibility="collapsed",
            key="lab_escala",
        )

    seleccion_ubicacion = None

    with escala_columnas[1]:
        if escala == "Por estado":
            columna_ubicacion = (
                "ENTIDAD_FEDERATIVA_ANUAL"
            )

            opciones_ubicacion = (
                sorted(
                    base[
                        columna_ubicacion
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if (
                    columna_ubicacion
                    in base.columns
                )
                else []
            )

            st.markdown(
                '<span class="lab-chip lab-ubicacion">Estado</span>',
                unsafe_allow_html=True,
            )

            if not opciones_ubicacion:
                st.warning(
                    "La columna de entidad federativa existe, "
                    "pero todavía no contiene datos recuperados."
                )
                return

            seleccion_ubicacion = st.selectbox(
                "Estado",
                opciones_ubicacion,
                label_visibility="collapsed",
                key="lab_estado",
            )

        elif escala == "Por institución o centro":
            columna_ubicacion = (
                "INSTITUCION_ANUAL"
            )

            opciones_ubicacion = (
                sorted(
                    base[
                        columna_ubicacion
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if (
                    columna_ubicacion
                    in base.columns
                )
                else []
            )

            st.markdown(
                '<span class="lab-chip lab-ubicacion">'
                'Universidad o centro de investigación'
                '</span>',
                unsafe_allow_html=True,
            )

            if not opciones_ubicacion:
                st.warning(
                    "La columna de institución existe, pero "
                    "todavía no contiene datos recuperados."
                )
                return

            seleccion_ubicacion = st.selectbox(
                "Institución",
                opciones_ubicacion,
                label_visibility="collapsed",
                key="lab_institucion",
            )

        else:
            st.markdown(
                '<span class="lab-chip lab-ubicacion">'
                'Estados Unidos Mexicanos'
                '</span>',
                unsafe_allow_html=True,
            )

    base_escala = filtrar_escala_laboratorio(
        base,
        escala,
        seleccion_ubicacion,
    )

    # --------------------------------------------------------
    # OBJETIVO Y TIEMPO
    # --------------------------------------------------------

    fila_objetivo = st.columns(
        [
            1.6,
            1,
        ]
    )

    with fila_objetivo[0]:
        st.markdown(
            '<span class="lab-chip lab-objetivo">Quiero analizar</span>',
            unsafe_allow_html=True,
        )

        objetivo = st.selectbox(
            "Objetivo",
            [
                "la distribución de",
                "la evolución de",
                "la relación entre",
                "la comparación de",
            ],
            label_visibility="collapsed",
            key="lab_objetivo",
        )

    with fila_objetivo[1]:
        st.markdown(
            '<span class="lab-chip lab-tiempo">Tiempo</span>',
            unsafe_allow_html=True,
        )

        usa_tiempo = st.toggle(
            "Que dependa del tiempo",
            value=(
                objetivo == "la evolución de"
            ),
            key="lab_tiempo",
        )

    # --------------------------------------------------------
    # VARIABLES POR FAMILIA
    # --------------------------------------------------------

    if not variables_disponibles:
        st.warning(
            "No existen variables con datos disponibles."
        )
        return

    familias = sorted(
        inventario.loc[
            inventario["CON_DATOS"]
            & inventario["TIPO"].ne(
                "identificador"
            ),
            "FAMILIA",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    familia = st.selectbox(
        "Familia de la variable principal",
        familias,
        key="lab_familia_1",
    )

    opciones_familia = (
        inventario.loc[
            inventario["CON_DATOS"]
            & inventario["FAMILIA"].eq(
                familia
            )
            & inventario["TIPO"].ne(
                "identificador"
            ),
            "VARIABLE",
        ]
        .sort_values()
        .tolist()
    )

    columnas_variables = st.columns(3)

    with columnas_variables[0]:
        st.markdown(
            '<span class="lab-chip lab-variable">Variable principal</span>',
            unsafe_allow_html=True,
        )

        variable_1 = st.selectbox(
            "Variable principal",
            opciones_familia,
            label_visibility="collapsed",
            key="lab_variable_1",
        )

    opciones_restantes = [
        variable
        for variable in variables_disponibles
        if variable != variable_1
    ]

    with columnas_variables[1]:
        st.markdown(
            '<span class="lab-chip lab-variable">Variable secundaria</span>',
            unsafe_allow_html=True,
        )

        variable_2 = st.selectbox(
            "Variable secundaria",
            [
                "Ninguna",
                *opciones_restantes,
            ],
            label_visibility="collapsed",
            key="lab_variable_2",
        )

    seleccionadas = [
        variable_1,
    ]

    if variable_2 != "Ninguna":
        seleccionadas.append(
            variable_2
        )

    opciones_tercera = [
        variable
        for variable in variables_disponibles
        if variable not in seleccionadas
    ]

    with columnas_variables[2]:
        st.markdown(
            '<span class="lab-chip lab-variable">Tercera variable</span>',
            unsafe_allow_html=True,
        )

        variable_3 = st.selectbox(
            "Tercera variable",
            [
                "Ninguna",
                *opciones_tercera,
            ],
            label_visibility="collapsed",
            key="lab_variable_3",
        )

    if variable_3 != "Ninguna":
        seleccionadas.append(
            variable_3
        )

    # --------------------------------------------------------
    # PERIODO
    # --------------------------------------------------------

    años_disponibles = sorted(
        base_escala[
            "AÑO"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if usa_tiempo:
        periodo = st.slider(
            "Periodo",
            min_value=min(
                años_disponibles
            ),
            max_value=max(
                años_disponibles
            ),
            value=(
                min(
                    años_disponibles
                ),
                max(
                    años_disponibles
                ),
            ),
            key="lab_periodo",
        )

        anio = None

        texto_tiempo = (
            f'<span class="lab-chip lab-tiempo">'
            f'{periodo[0]}–{periodo[1]}</span>'
        )

        frase_tiempo = (
            "y observar cómo cambia durante"
        )

    else:
        anio = st.selectbox(
            "Año de referencia",
            años_disponibles,
            index=len(
                años_disponibles
            ) - 1,
            key="lab_anio",
        )

        periodo = None

        texto_tiempo = (
            f'<span class="lab-chip lab-tiempo">'
            f'{anio}</span>'
        )

        frase_tiempo = (
            "en el año"
        )

    variables_html = " y ".join(
        [
            (
                '<span class="lab-chip lab-variable">'
                f'{variable}'
                '</span>'
            )
            for variable in seleccionadas
        ]
    )

    ubicacion_texto = (
        seleccion_ubicacion
        if seleccion_ubicacion is not None
        else "México"
    )

    st.markdown(
        (
            '<div class="lab-sentence">'
            'Me gustaría analizar '
            f'<span class="lab-chip lab-objetivo">{objetivo}</span> '
            f'{variables_html} '
            f'{frase_tiempo} {texto_tiempo}, '
            'a escala '
            f'<span class="lab-chip lab-ubicacion">'
            f'{ubicacion_texto}'
            '</span>.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RECOMENDACIÓN
    # --------------------------------------------------------

    recomendaciones = recomendar_visualizaciones(
        seleccionadas,
        usa_tiempo,
    )

    if not recomendaciones:
        st.warning(
            "La combinación seleccionada requiere otro "
            "tipo de tratamiento antes de graficarse."
        )
        return

    st.subheader(
        "Recomendación del sistema"
    )

    tarjetas = st.columns(
        min(
            3,
            len(
                recomendaciones
            ),
        )
    )

    for indice, recomendacion in enumerate(
        recomendaciones[:3]
    ):
        with tarjetas[indice]:
            st.markdown(
                f"**{recomendacion['grafica']}**"
            )

            st.markdown(
                (
                    '<div class="lab-score">'
                    f"{recomendacion['idoneidad']}%"
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            st.progress(
                recomendacion[
                    "idoneidad"
                ] / 100
            )

            st.caption(
                recomendacion[
                    "razon"
                ]
            )

    opciones_grafica = [
        recomendacion["grafica"]
        for recomendacion in recomendaciones
        if recomendacion["idoneidad"] >= 50
    ]

    tipo_grafica = st.radio(
        "Tipo de gráfica",
        opciones_grafica,
        horizontal=True,
        key="lab_grafica",
    )

    if not st.button(
        "Generar visualización",
        type="primary",
        width="stretch",
        key="lab_generar",
    ):
        return

    try:
        (
            base_analisis,
            columnas,
            tipos,
        ) = construir_dataset_laboratorio(
            base_escala,
            seleccionadas,
            usa_tiempo,
            periodo,
            anio,
        )

        if base_analisis.empty:
            st.warning(
                "No existen registros completos para la "
                "combinación seleccionada."
            )
            return

        figura, datos = generar_figura_laboratorio(
            base_analisis,
            seleccionadas,
            columnas,
            tipos,
            usa_tiempo,
            tipo_grafica,
        )

        figura.update_layout(
            height=540,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
            hovermode=(
                "x unified"
                if usa_tiempo
                else "closest"
            ),
        )

        st.plotly_chart(
            figura,
            width="stretch",
            key="lab_resultado",
        )

        descripcion, interpretacion = (
            interpretar_laboratorio(
                base_analisis,
                seleccionadas,
                columnas,
                tipos,
                escala,
            )
        )

        izquierda, derecha = st.columns(2)

        with izquierda:
            st.subheader(
                "Descripción de la gráfica"
            )
            st.write(
                descripcion
            )

        with derecha:
            st.subheader(
                "Interpretación automática"
            )
            st.write(
                interpretacion
            )

        if (
            usa_tiempo
            and periodo is not None
            and periodo[1] >= 2025
        ):
            st.warning(
                "El año 2025 debe interpretarse con cautela "
                "hasta confirmar la cobertura completa de la fuente."
            )

        with st.expander(
            "Ver datos utilizados"
        ):
            st.dataframe(
                datos,
                width="stretch",
                hide_index=True,
            )

    except Exception as error:
        st.error(
            "No fue posible generar la visualización: "
            f"{error}"
        )


# ============================================================
# APLICACIÓN
# ============================================================

def main() -> None:
    st.title("SNII Insight")
    st.markdown(
        '<p class="snii-subtitle">'
        "Plataforma para explorar la evolución histórica del "
        "Sistema Nacional de Investigadoras e Investigadores."
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        df, fuente = obtener_base()

    except Exception as error:
        st.error(f"No fue posible cargar la base: {error}")
        st.stop()

    st.sidebar.title("SNII Insight")
    st.sidebar.caption(f"Fuente activa: {fuente}")
    st.sidebar.metric("Filas persona-año", f"{len(df):,}")
    st.sidebar.metric(
        "Personas únicas",
        f"{df['ID_PERSONA_EXACTA'].nunique():,}",
    )

    modulo = st.sidebar.radio(
        "Módulo",
        [
            "Panorama actual",
            "Proyecciones",
            "Historial del investigador",
            "Laboratorio de visualización",
        ],
    )

    if modulo == "Panorama actual":
        render_panorama(df)

    elif modulo == "Proyecciones":
        render_proyecciones(df)

    elif modulo == "Historial del investigador":
        render_investigador(df)

    else:
        render_laboratorio_visualizacion(df)


if __name__ == "__main__":
    main()
