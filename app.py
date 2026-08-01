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
        columnas_disponibles = pd.read_parquet(path, engine="pyarrow").columns
        columnas = [
            col for col in COLUMNAS_ANALITICAS if col in columnas_disponibles
        ]
        df = pd.read_parquet(path, columns=columnas, engine="pyarrow")

    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(
            path,
            sheet_name="PERSONA_AÑO",
            engine="openpyxl",
        )
        columnas = [col for col in COLUMNAS_ANALITICAS if col in df.columns]
        df = df[columnas].copy()

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

    columnas = [col for col in COLUMNAS_ANALITICAS if col in df.columns]
    return preparar_base(df[columnas].copy())


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

VARIABLES_LABORATORIO = {
    "Sexo": {
        "columna": "SEXO_CONSOLIDADO",
        "tipo": "categorica",
    },
    "Nivel SNII": {
        "columna": "NIVEL_SNII_ETIQUETA",
        "tipo": "categorica",
    },
    "STEM / No STEM": {
        "columna": "GRUPO_STEM_BINARIO",
        "tipo": "categorica",
    },
    "Clasificación académica": {
        "columna": "CLASIFICACION_STEM_ANUAL",
        "tipo": "categorica",
    },
    "Área del conocimiento": {
        "columna": "AREA_DEL_CONOCIMIENTO_ANUAL",
        "tipo": "categorica",
    },
    "Disciplina": {
        "columna": "DISCIPLINA_ANUAL",
        "tipo": "categorica",
    },
    "Años con registro": {
        "columna": "NUMERO_AÑOS_PRESENTE",
        "tipo": "numerica",
    },
    "Antigüedad observada": {
        "columna": "_ANTIGUEDAD_DERIVADA",
        "tipo": "numerica",
    },
    "Completitud del expediente": {
        "columna": "PORCENTAJE_COMPLETITUD_CLAVE",
        "tipo": "numerica",
    },
}


def preparar_base_laboratorio(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Crea variables derivadas para el laboratorio."""

    base = df.copy()

    if (
        "PRIMER_AÑO" in base.columns
        and "AÑO" in base.columns
    ):
        primer_anio = pd.to_numeric(
            base["PRIMER_AÑO"],
            errors="coerce",
        )

        anio = pd.to_numeric(
            base["AÑO"],
            errors="coerce",
        )

        base["_ANTIGUEDAD_DERIVADA"] = (
            anio - primer_anio + 1
        ).clip(lower=1)

    else:
        base["_ANTIGUEDAD_DERIVADA"] = np.nan

    for columna in [
        "NUMERO_AÑOS_PRESENTE",
        "PORCENTAJE_COMPLETITUD_CLAVE",
        "_ANTIGUEDAD_DERIVADA",
    ]:
        if columna in base.columns:
            base[columna] = pd.to_numeric(
                base[columna],
                errors="coerce",
            )

    return base


def variables_disponibles_laboratorio(
    df: pd.DataFrame,
) -> list[str]:
    """Devuelve variables cuyo campo existe y tiene datos."""

    disponibles = []

    for nombre, especificacion in (
        VARIABLES_LABORATORIO.items()
    ):
        columna = especificacion["columna"]

        if columna in df.columns and df[columna].notna().any():
            disponibles.append(nombre)

    return disponibles


def recomendar_visualizaciones(
    objetivo: str,
    variables: list[str],
    usa_tiempo: bool,
) -> list[dict[str, object]]:
    """Motor de recomendación determinista y explicable."""

    tipos = [
        VARIABLES_LABORATORIO[var]["tipo"]
        for var in variables
    ]

    n_categoricas = tipos.count("categorica")
    n_numericas = tipos.count("numerica")

    recomendaciones: dict[str, tuple[int, str]] = {}

    def agregar(
        grafica: str,
        puntuacion: int,
        razon: str,
    ) -> None:
        puntuacion = max(
            0,
            min(
                100,
                puntuacion,
            ),
        )

        actual = recomendaciones.get(grafica)

        if actual is None or puntuacion > actual[0]:
            recomendaciones[grafica] = (
                puntuacion,
                razon,
            )

    if usa_tiempo:
        if n_categoricas == 1 and n_numericas == 0:
            agregar(
                "Barras apiladas",
                96,
                "Compara la composición de una variable categórica "
                "a través de los años.",
            )
            agregar(
                "Líneas múltiples",
                88,
                "Permite seguir con claridad la tendencia de cada categoría.",
            )
            agregar(
                "Área apilada",
                80,
                "Destaca la evolución del total y su composición.",
            )
            agregar(
                "Dona",
                20,
                "Una dona resume un solo momento y no representa bien el tiempo.",
            )

        elif n_categoricas >= 2 and n_numericas == 0:
            agregar(
                "Barras apiladas",
                90,
                "Resume dos categorías por año, aunque puede requerir "
                "limitar el número de grupos.",
            )
            agregar(
                "Líneas múltiples",
                76,
                "Es útil si el número de combinaciones es reducido.",
            )
            agregar(
                "Mapa de calor",
                86,
                "Facilita observar patrones entre categorías y años.",
            )

        elif n_numericas >= 1 and n_categoricas >= 1:
            agregar(
                "Boxplot por periodo",
                92,
                "Compara la distribución numérica entre categorías y años.",
            )
            agregar(
                "Líneas de mediana",
                84,
                "Resume la tendencia central de cada grupo en el tiempo.",
            )

        elif n_numericas >= 1:
            agregar(
                "Línea temporal",
                94,
                "Muestra la evolución anual de la variable numérica.",
            )
            agregar(
                "Área",
                76,
                "Ayuda a enfatizar la magnitud acumulada visual.",
            )

    else:
        if n_categoricas == 1 and n_numericas == 0:
            agregar(
                "Dona",
                96,
                "Muestra de manera clara la composición de pocas categorías.",
            )
            agregar(
                "Barras",
                91,
                "Facilita comparar con precisión el tamaño de cada grupo.",
            )

        elif n_categoricas >= 2 and n_numericas == 0:
            agregar(
                "Barras apiladas",
                93,
                "Compara la composición conjunta de dos variables categóricas.",
            )
            agregar(
                "Mapa de calor",
                88,
                "Permite identificar concentraciones entre categorías.",
            )
            agregar(
                "Dona",
                28,
                "No representa adecuadamente la relación entre dos categorías.",
            )

        elif n_numericas == 1 and n_categoricas == 0:
            agregar(
                "Histograma",
                97,
                "Representa la distribución de una variable numérica.",
            )
            agregar(
                "Boxplot",
                89,
                "Resume mediana, dispersión y valores atípicos.",
            )

        elif n_numericas >= 1 and n_categoricas >= 1:
            agregar(
                "Boxplot",
                96,
                "Compara la distribución numérica entre grupos.",
            )
            agregar(
                "Violin plot",
                88,
                "Muestra forma y densidad de la distribución por grupo.",
            )
            agregar(
                "Burbujas",
                72,
                "Puede resumir tres dimensiones, pero requiere agregación.",
            )

        elif n_numericas >= 2:
            agregar(
                "Dispersión",
                96,
                "Muestra la relación entre dos variables numéricas.",
            )
            agregar(
                "Burbujas",
                88,
                "Añade una tercera medida mediante el tamaño del marcador.",
            )

    if objetivo == "la distribución de":
        for nombre in [
            "Dona",
            "Histograma",
            "Barras",
            "Boxplot",
        ]:
            if nombre in recomendaciones:
                score, razon = recomendaciones[nombre]
                recomendaciones[nombre] = (
                    min(100, score + 3),
                    razon,
                )

    if objetivo == "la evolución de":
        for nombre in [
            "Barras apiladas",
            "Líneas múltiples",
            "Línea temporal",
            "Área apilada",
        ]:
            if nombre in recomendaciones:
                score, razon = recomendaciones[nombre]
                recomendaciones[nombre] = (
                    min(100, score + 3),
                    razon,
                )

    salida = [
        {
            "grafica": grafica,
            "idoneidad": puntuacion,
            "razon": razon,
        }
        for grafica, (
            puntuacion,
            razon,
        ) in recomendaciones.items()
    ]

    return sorted(
        salida,
        key=lambda item: item["idoneidad"],
        reverse=True,
    )


def construir_dataset_laboratorio(
    df: pd.DataFrame,
    variables: list[str],
    usa_tiempo: bool,
    periodo: tuple[int, int] | None,
    anio: int | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filtra y prepara las columnas seleccionadas."""

    base = preparar_base_laboratorio(df)

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
            base["AÑO"].eq(anio)
        ].copy()

    columnas = [
        VARIABLES_LABORATORIO[var]["columna"]
        for var in variables
    ]

    columnas_presentes = [
        columna
        for columna in columnas
        if columna in base.columns
    ]

    tipos = [
        VARIABLES_LABORATORIO[var]["tipo"]
        for var in variables
    ]

    columnas_requeridas = [
        "ID_PERSONA_EXACTA",
        "AÑO",
        *columnas_presentes,
    ]

    base = (
        base[columnas_requeridas]
        .dropna(
            subset=columnas_presentes,
            how="any",
        )
        .copy()
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
    """Genera la visualización seleccionada."""

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
        if tipo == "categorica"
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

        fig = px.pie(
            resumen,
            names=columna,
            values="PERSONAS",
            hole=0.58,
            labels={
                columna: nombre,
                "PERSONAS": "Personas",
            },
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        return fig, resumen

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

        fig = px.bar(
            resumen,
            x=columna,
            y="PERSONAS",
            labels={
                columna: nombre,
                "PERSONAS": "Personas",
            },
        )

        return fig, resumen

    if grafica in {
        "Barras apiladas",
        "Área apilada",
        "Líneas múltiples",
    }:
        if usa_tiempo:
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

            if len(categoricas) >= 2:
                _, segunda_columna = categoricas[1]

                top = (
                    base.groupby(segunda_columna)[
                        "ID_PERSONA_EXACTA"
                    ]
                    .nunique()
                    .nlargest(8)
                    .index
                )

                base_reducida = base.loc[
                    base[segunda_columna].isin(top)
                ]

                resumen = (
                    base_reducida.groupby(
                        [
                            "AÑO",
                            segunda_columna,
                            columna_color,
                        ]
                    )["ID_PERSONA_EXACTA"]
                    .nunique()
                    .rename("PERSONAS")
                    .reset_index()
                )

            if grafica == "Barras apiladas":
                fig = px.bar(
                    resumen,
                    x="AÑO",
                    y="PERSONAS",
                    color=columna_color,
                    barmode="stack",
                    facet_row=(
                        categoricas[1][1]
                        if len(categoricas) >= 2
                        else None
                    ),
                    labels={
                        "PERSONAS": "Personas",
                        columna_color: nombre_color,
                    },
                )

            elif grafica == "Área apilada":
                fig = px.area(
                    resumen,
                    x="AÑO",
                    y="PERSONAS",
                    color=columna_color,
                    groupnorm=None,
                    labels={
                        "PERSONAS": "Personas",
                        columna_color: nombre_color,
                    },
                )

            else:
                fig = px.line(
                    resumen,
                    x="AÑO",
                    y="PERSONAS",
                    color=columna_color,
                    markers=True,
                    labels={
                        "PERSONAS": "Personas",
                        columna_color: nombre_color,
                    },
                )

            return fig, resumen

        nombre_x, columna_x = categoricas[0]

        if len(categoricas) >= 2:
            nombre_color, columna_color = categoricas[1]

            resumen = (
                base.groupby(
                    [
                        columna_x,
                        columna_color,
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            fig = px.bar(
                resumen,
                x=columna_x,
                y="PERSONAS",
                color=columna_color,
                barmode="stack",
                labels={
                    columna_x: nombre_x,
                    columna_color: nombre_color,
                    "PERSONAS": "Personas",
                },
            )

        else:
            resumen = (
                base.groupby(columna_x)["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            fig = px.bar(
                resumen,
                x=columna_x,
                y="PERSONAS",
                labels={
                    columna_x: nombre_x,
                    "PERSONAS": "Personas",
                },
            )

        return fig, resumen

    if grafica in {
        "Histograma",
    }:
        nombre, columna = numericas[0]

        fig = px.histogram(
            base,
            x=columna,
            nbins=30,
            labels={
                columna: nombre,
                "count": "Registros",
            },
        )

        return fig, base[[columna]].copy()

    if grafica in {
        "Boxplot",
        "Boxplot por periodo",
        "Violin plot",
    }:
        nombre_num, columna_num = numericas[0]

        if categoricas:
            nombre_cat, columna_cat = categoricas[0]
        elif usa_tiempo:
            nombre_cat, columna_cat = "Año", "AÑO"
        else:
            nombre_cat, columna_cat = "Total", None

        color = (
            categoricas[1][1]
            if len(categoricas) >= 2
            else None
        )

        if grafica == "Violin plot":
            fig = px.violin(
                base,
                x=columna_cat,
                y=columna_num,
                color=color,
                box=True,
                points=False,
                labels={
                    columna_num: nombre_num,
                    columna_cat: nombre_cat,
                },
            )

        else:
            fig = px.box(
                base,
                x=columna_cat,
                y=columna_num,
                color=color,
                points="outliers",
                labels={
                    columna_num: nombre_num,
                    columna_cat: nombre_cat,
                },
            )

        return fig, base[
            [
                columna
                for columna in [
                    columna_cat,
                    columna_num,
                    color,
                ]
                if columna is not None
            ]
        ].copy()

    if grafica in {
        "Mapa de calor",
    }:
        if usa_tiempo:
            nombre_cat, columna_cat = categoricas[0]

            resumen = (
                base.groupby(
                    [
                        columna_cat,
                        "AÑO",
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            pivote = resumen.pivot(
                index=columna_cat,
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

            pivote = resumen.pivot(
                index=columna_1,
                columns=columna_2,
                values="PERSONAS",
            ).fillna(0)

        fig = px.imshow(
            pivote,
            aspect="auto",
            labels={
                "color": "Personas",
            },
        )

        return fig, resumen

    if grafica in {
        "Dispersión",
        "Burbujas",
    }:
        if len(numericas) < 2:
            nombre_num, columna_num = numericas[0]
            nombre_cat, columna_cat = categoricas[0]

            resumen = (
                base.groupby(columna_cat)
                .agg(
                    VALOR_MEDIO=(
                        columna_num,
                        "mean",
                    ),
                    PERSONAS=(
                        "ID_PERSONA_EXACTA",
                        "nunique",
                    ),
                )
                .reset_index()
            )

            fig = px.scatter(
                resumen,
                x="PERSONAS",
                y="VALOR_MEDIO",
                size=(
                    "PERSONAS"
                    if grafica == "Burbujas"
                    else None
                ),
                color=columna_cat,
                hover_name=columna_cat,
                labels={
                    "PERSONAS": "Personas",
                    "VALOR_MEDIO":
                        f"Promedio de {nombre_num}",
                },
                size_max=55,
            )

        else:
            nombre_x, columna_x = numericas[0]
            nombre_y, columna_y = numericas[1]

            color = (
                categoricas[0][1]
                if categoricas
                else None
            )

            resumen = base[
                [
                    columna
                    for columna in [
                        columna_x,
                        columna_y,
                        color,
                    ]
                    if columna is not None
                ]
            ].copy()

            fig = px.scatter(
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

        return fig, resumen

    if grafica in {
        "Línea temporal",
        "Líneas de mediana",
        "Área",
    }:
        nombre_num, columna_num = numericas[0]

        agrupadores = ["AÑO"]

        if categoricas:
            agrupadores.append(
                categoricas[0][1]
            )

        resumen = (
            base.groupby(agrupadores)[columna_num]
            .median()
            .rename("MEDIANA")
            .reset_index()
        )

        color = (
            categoricas[0][1]
            if categoricas
            else None
        )

        if grafica == "Área":
            fig = px.area(
                resumen,
                x="AÑO",
                y="MEDIANA",
                color=color,
                labels={
                    "MEDIANA": f"Mediana de {nombre_num}",
                },
            )

        else:
            fig = px.line(
                resumen,
                x="AÑO",
                y="MEDIANA",
                color=color,
                markers=True,
                labels={
                    "MEDIANA": f"Mediana de {nombre_num}",
                },
            )

        return fig, resumen

    raise ValueError(
        f"El tipo de gráfica '{grafica}' aún no está implementado."
    )


def interpretar_resultado_laboratorio(
    base: pd.DataFrame,
    variables: list[str],
    columnas: list[str],
    tipos: list[str],
    usa_tiempo: bool,
) -> tuple[str, str]:
    """Genera una descripción e interpretación breve basada en datos."""

    descripcion = (
        "La visualización resume personas únicas del SNII "
        "según las variables seleccionadas."
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
        if tipo == "categorica"
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

    if usa_tiempo and categoricas:
        nombre, columna = categoricas[0]

        resumen = (
            base.groupby(
                [
                    "AÑO",
                    columna,
                ]
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .rename("PERSONAS")
            .reset_index()
        )

        anios = sorted(
            resumen["AÑO"].dropna().unique()
        )

        if len(anios) >= 2:
            primero = anios[0]
            ultimo = anios[-1]

            inicio = resumen.loc[
                resumen["AÑO"].eq(primero)
            ]

            final = resumen.loc[
                resumen["AÑO"].eq(ultimo)
            ]

            categoria_final = (
                final.sort_values(
                    "PERSONAS",
                    ascending=False,
                )
                .iloc[0]
            )

            interpretacion = (
                f"En {ultimo}, la categoría con mayor presencia en "
                f"{nombre.lower()} fue "
                f"'{categoria_final[columna]}', con "
                f"{int(categoria_final['PERSONAS']):,} personas. "
                f"La gráfica permite comparar este patrón con {primero} "
                "y reconocer cambios de composición durante el periodo."
            )

            return descripcion, interpretacion

    if categoricas and not numericas:
        nombre, columna = categoricas[0]

        resumen = (
            base.groupby(columna)["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        if not resumen.empty:
            total = resumen.sum()
            principal = resumen.index[0]
            cantidad = int(resumen.iloc[0])
            porcentaje = (
                cantidad / total * 100
                if total
                else 0
            )

            interpretacion = (
                f"La categoría con mayor representación en "
                f"{nombre.lower()} es '{principal}', con "
                f"{cantidad:,} personas ({porcentaje:.1f}% del total "
                "clasificado)."
            )

            return descripcion, interpretacion

    if numericas and categoricas:
        nombre_num, columna_num = numericas[0]
        nombre_cat, columna_cat = categoricas[0]

        resumen = (
            base.groupby(columna_cat)[columna_num]
            .median()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        if not resumen.empty:
            mayor = resumen.index[0]
            menor = resumen.index[-1]

            interpretacion = (
                f"La mediana más alta de {nombre_num.lower()} "
                f"corresponde a '{mayor}' "
                f"({resumen.iloc[0]:.1f}), mientras que la más baja "
                f"se observa en '{menor}' ({resumen.iloc[-1]:.1f})."
            )

            return descripcion, interpretacion

    if len(numericas) >= 2:
        nombre_x, columna_x = numericas[0]
        nombre_y, columna_y = numericas[1]

        pares = base[
            [
                columna_x,
                columna_y,
            ]
        ].dropna()

        correlacion = pares[
            columna_x
        ].corr(
            pares[columna_y],
            method="spearman",
        )

        if pd.notna(correlacion):
            intensidad = (
                "fuerte"
                if abs(correlacion) >= 0.70
                else "moderada"
                if abs(correlacion) >= 0.40
                else "débil"
            )

            direccion = (
                "positiva"
                if correlacion > 0
                else "negativa"
            )

            interpretacion = (
                f"La relación de Spearman entre "
                f"{nombre_x.lower()} y {nombre_y.lower()} es "
                f"{direccion} y {intensidad} "
                f"(ρ = {correlacion:.2f})."
            )

            return descripcion, interpretacion

    if numericas:
        nombre, columna = numericas[0]

        valores = base[columna].dropna()

        if not valores.empty:
            interpretacion = (
                f"La mediana de {nombre.lower()} es "
                f"{valores.median():.1f}; el 50% central de los "
                f"registros se encuentra entre "
                f"{valores.quantile(0.25):.1f} y "
                f"{valores.quantile(0.75):.1f}."
            )

            return descripcion, interpretacion

    return (
        descripcion,
        "La selección permite explorar diferencias y patrones. "
        "La interpretación debe complementarse con la revisión "
        "de cobertura y calidad de los datos.",
    )


def render_laboratorio_visualizacion(
    df: pd.DataFrame,
) -> None:
    """Constructor guiado de preguntas, gráficas e interpretación."""

    st.header(
        "4. Laboratorio de visualización"
    )

    st.markdown(
        '<p class="snii-subtitle">'
        "Construye tu pregunta y deja que SNII Insight "
        "te recomiende la visualización más adecuada."
        "</p>",
        unsafe_allow_html=True,
    )

    base_laboratorio = preparar_base_laboratorio(
        df
    )

    variables_disponibles = (
        variables_disponibles_laboratorio(
            base_laboratorio
        )
    )

    if not variables_disponibles:
        st.warning(
            "No se encontraron variables disponibles para "
            "construir el laboratorio."
        )
        return

    st.markdown(
        """
        <span class="lab-chip lab-objetivo">Objetivo</span>
        <span class="lab-chip lab-variable">Variable</span>
        <span class="lab-chip lab-tiempo">Tiempo</span>
        <span class="lab-chip lab-filtro">Filtro</span>
        <span class="lab-chip lab-ubicacion">Ubicación</span>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Construye tu pregunta"
    )

    columna_objetivo, columna_tiempo = st.columns(
        [
            1.5,
            1,
        ]
    )

    with columna_objetivo:
        st.markdown(
            '<span class="lab-chip lab-objetivo">¿Qué deseas analizar?</span>',
            unsafe_allow_html=True,
        )

        objetivo = st.selectbox(
            "Objetivo analítico",
            [
                "la distribución de",
                "la evolución de",
                "la relación entre",
                "la comparación de",
            ],
            label_visibility="collapsed",
            key="lab_objetivo",
        )

    with columna_tiempo:
        st.markdown(
            '<span class="lab-chip lab-tiempo">Dimensión temporal</span>',
            unsafe_allow_html=True,
        )

        usa_tiempo = st.toggle(
            "Observar cambios en el tiempo",
            value=(
                objetivo == "la evolución de"
            ),
            key="lab_usa_tiempo",
        )

    columnas_variables = st.columns(3)

    variables_seleccionadas = []

    with columnas_variables[0]:
        st.markdown(
            '<span class="lab-chip lab-variable">Variable principal</span>',
            unsafe_allow_html=True,
        )

        variable_1 = st.selectbox(
            "Variable principal",
            variables_disponibles,
            label_visibility="collapsed",
            key="lab_variable_1",
        )

        variables_seleccionadas.append(
            variable_1
        )

    opciones_secundarias = [
        "Ninguna",
        *[
            variable
            for variable in variables_disponibles
            if variable != variable_1
        ],
    ]

    with columnas_variables[1]:
        st.markdown(
            '<span class="lab-chip lab-variable">Variable secundaria</span>',
            unsafe_allow_html=True,
        )

        variable_2 = st.selectbox(
            "Variable secundaria",
            opciones_secundarias,
            label_visibility="collapsed",
            key="lab_variable_2",
        )

        if variable_2 != "Ninguna":
            variables_seleccionadas.append(
                variable_2
            )

    opciones_terciarias = [
        "Ninguna",
        *[
            variable
            for variable in variables_disponibles
            if variable not in variables_seleccionadas
        ],
    ]

    with columnas_variables[2]:
        st.markdown(
            '<span class="lab-chip lab-variable">Tercera variable</span>',
            unsafe_allow_html=True,
        )

        variable_3 = st.selectbox(
            "Tercera variable",
            opciones_terciarias,
            label_visibility="collapsed",
            key="lab_variable_3",
        )

        if variable_3 != "Ninguna":
            variables_seleccionadas.append(
                variable_3
            )

    años_disponibles = sorted(
        base_laboratorio[
            "AÑO"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if usa_tiempo:
        st.markdown(
            '<span class="lab-chip lab-tiempo">Periodo</span>',
            unsafe_allow_html=True,
        )

        periodo = st.slider(
            "Periodo de análisis",
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

        anio_seleccionado = None

    else:
        st.markdown(
            '<span class="lab-chip lab-tiempo">Año de referencia</span>',
            unsafe_allow_html=True,
        )

        anio_seleccionado = st.selectbox(
            "Año",
            años_disponibles,
            index=len(
                años_disponibles
            ) - 1,
            label_visibility="collapsed",
            key="lab_anio",
        )

        periodo = None

    texto_variables = (
        " y ".join(
            [
                (
                    f'<span class="lab-chip lab-variable">'
                    f'{variable}</span>'
                )
                for variable in variables_seleccionadas
            ]
        )
    )

    if usa_tiempo and periodo is not None:
        texto_tiempo = (
            f'<span class="lab-chip lab-tiempo">'
            f'{periodo[0]}–{periodo[1]}</span>'
        )

        complemento_tiempo = (
            "y observar cómo cambia durante"
        )

    else:
        texto_tiempo = (
            f'<span class="lab-chip lab-tiempo">'
            f'{anio_seleccionado}</span>'
        )

        complemento_tiempo = (
            "en el año"
        )

    st.markdown(
        (
            '<div class="lab-sentence">'
            'Me gustaría analizar '
            f'<span class="lab-chip lab-objetivo">{objetivo}</span> '
            f'{texto_variables} '
            f'{complemento_tiempo} {texto_tiempo}.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    recomendaciones = recomendar_visualizaciones(
        objetivo,
        variables_seleccionadas,
        usa_tiempo,
    )

    if not recomendaciones:
        st.warning(
            "La combinación seleccionada todavía no tiene "
            "una recomendación disponible."
        )
        return

    st.subheader(
        "Recomendación del sistema"
    )

    principal = recomendaciones[0]

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
                recomendacion["idoneidad"]
                / 100
            )

            st.caption(
                recomendacion["razon"]
            )

    opciones_grafica = [
        recomendacion["grafica"]
        for recomendacion in recomendaciones
        if recomendacion["idoneidad"] >= 50
    ]

    if not opciones_grafica:
        opciones_grafica = [
            principal["grafica"]
        ]

    st.markdown(
        '<span class="lab-chip lab-filtro">Tipo de gráfica</span>',
        unsafe_allow_html=True,
    )

    tipo_grafica = st.radio(
        "Selecciona el tipo de gráfica",
        opciones_grafica,
        horizontal=True,
        label_visibility="collapsed",
        key="lab_tipo_grafica",
    )

    generar = st.button(
        "Generar visualización",
        type="primary",
        width="stretch",
        key="lab_generar",
    )

    if not generar:
        return

    try:
        (
            base_analisis,
            columnas,
            tipos,
        ) = construir_dataset_laboratorio(
            base_laboratorio,
            variables_seleccionadas,
            usa_tiempo,
            periodo,
            anio_seleccionado,
        )

        if base_analisis.empty:
            st.warning(
                "No existen registros completos para la "
                "combinación seleccionada."
            )
            return

        figura, datos_grafica = (
            generar_figura_laboratorio(
                base_analisis,
                variables_seleccionadas,
                columnas,
                tipos,
                usa_tiempo,
                tipo_grafica,
            )
        )

        figura.update_layout(
            height=520,
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
            key="lab_grafica_resultado",
        )

        descripcion, interpretacion = (
            interpretar_resultado_laboratorio(
                base_analisis,
                variables_seleccionadas,
                columnas,
                tipos,
                usa_tiempo,
            )
        )

        izquierda, derecha = st.columns(2)

        with izquierda:
            st.subheader(
                "¿Qué muestra esta gráfica?"
            )
            st.write(
                descripcion
            )

        with derecha:
            st.subheader(
                "¿Qué se identificó?"
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
                "hasta confirmar si la fuente representa el "
                "padrón nacional completo."
            )

        with st.expander(
            "Ver datos utilizados en la gráfica"
        ):
            st.dataframe(
                datos_grafica,
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
