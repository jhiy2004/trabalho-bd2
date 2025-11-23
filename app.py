import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard – Gráficos",
    layout="wide"
)

st.title("Dashboard – Séries temporais e indicadores")

st.markdown("""
Este dashboard é uma versão **interativa** do notebook `graficos.ipynb`,
utilizando **apenas** o arquivo `df_final.parquet`.

O foco aqui é:

1. Reproduzir os principais **gráficos de séries temporais**;
2. Mostrar **rankings de municípios** por indicador e ano;
3. Permitir **comparação justa entre indicadores** via normalização (min–max e z-score);
4. Comparar **várias cidades e tipos ao mesmo tempo**.
""")

# -------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------------------
@st.cache_data
def load_df_final(file) -> pd.DataFrame:
    """Carrega df_final.parquet a partir de upload."""
    return pd.read_parquet(file)


def preparar_df_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduz a lógica do graficos.ipynb:

    - Mantém 'Município' como id_vars
    - Derrete as demais colunas
    - Separa 'ano' e 'tipo' a partir do nome da coluna (ex: '2017_geral')
    - Garante que 'valor' é numérico
    - Cria colunas normalizadas por tipo (min–max e z-score)
    """
    if "Município" not in df.columns:
        raise ValueError("Coluna 'Município' não encontrada em df_final.parquet.")

    value_cols = [col for col in df.columns if col != "Município"]

    df_long = df.melt(
        id_vars="Município",
        value_vars=value_cols,
        var_name="ano_variavel",
        value_name="valor",
    )

    # Separa ano e tipo (ex: '2017_geral' -> ano=2017, tipo='geral')
    df_long[["ano", "tipo"]] = df_long["ano_variavel"].str.split("_", n=1, expand=True)
    df_long["ano"] = df_long["ano"].astype(int)

    # Garante que valor é numérico
    df_long["valor"] = pd.to_numeric(df_long["valor"], errors="coerce")

    # Normalização min–max por tipo
    def _minmax(s):
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    df_long["valor_minmax_tipo"] = (
        df_long.groupby("tipo")["valor"].transform(_minmax)
    )

    # Normalização z-score por tipo
    def _zscore(s):
        m, std = s.mean(), s.std()
        if pd.isna(m) or pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - m) / std

    df_long["valor_zscore_tipo"] = (
        df_long.groupby("tipo")["valor"].transform(_zscore)
    )

    return df_long


def plot_serie_temporal(df_plot, x_col, y_col, group_col, xlabel, ylabel, title):
    """Faz um lineplot simples com um grupo por linha."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for group, dados in df_plot.groupby(group_col):
        dados = dados.sort_values(x_col)
        ax.plot(dados[x_col], dados[y_col], marker="o", label=str(group))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    st.pyplot(fig)


# -------------------------------------------------------------------
# SIDEBAR – CARREGAMENTO DO df_final
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Dados")

file_df_final = st.sidebar.file_uploader(
    "Carregue o arquivo df_final.parquet",
    type=["parquet"]
)

if file_df_final is None:
    st.warning("Carregue o arquivo **df_final.parquet** na barra lateral para continuar.")
    st.stop()

# Carrega e prepara
df_final = load_df_final(file_df_final)

try:
    df_long = preparar_df_long(df_final)
except Exception as e:
    st.error(f"Erro ao preparar df_long: {e}")
    st.stop()

# Valores únicos
tipos_disponiveis = sorted(df_long["tipo"].unique())
municipios_disponiveis = sorted(df_long["Município"].unique())
anos_disponiveis = sorted(df_long["ano"].unique())

st.success(f"Dataset carregado com sucesso! Linhas: {df_final.shape[0]:,} | Colunas: {df_final.shape[1]}")

# ===================================================================
# SEÇÃO 1 – SÉRIE TEMPORAL POR TIPO (VÁRIOS MUNICÍPIOS)
# ===================================================================
st.markdown("---")
st.header("Série temporal por tipo — vários municípios")

st.markdown("""
Aqui escolhemos um **tipo de indicador** (por exemplo: `geral`, `covid`, `residentes`, `leitos`, `centros`, ...),
e observamos como os valores evoluem ao longo dos anos para vários municípios.

Isso responde:

- Quais municípios apresentam níveis mais altos nesse indicador;
- Se, em média, o indicador cresce, cai ou se mantém estável ao longo do tempo.
""")

col_sel_tipo, col_sel_top = st.columns([2, 1])

with col_sel_tipo:
    # tenta defaultar para "covid" ou "geral"
    default_tipo = 0
    for candidato in ["covid", "geral"]:
        if candidato in tipos_disponiveis:
            default_tipo = tipos_disponiveis.index(candidato)
            break

    tipo_escolhido = st.selectbox(
        "Escolha o tipo de indicador:",
        options=tipos_disponiveis,
        index=default_tipo
    )

with col_sel_top:
    limitar_municipios = st.checkbox(
        "Mostrar apenas os municípios com maior valor médio",
        value=True
    )
    top_n = 10
    if limitar_municipios:
        top_n = st.slider("Quantidade de municípios (Top N)", 3, 30, 10)

# Filtra por tipo
df_tipo = df_long[df_long["tipo"] == tipo_escolhido].copy()

# Se não sobrou nada numérico
if df_tipo["valor"].notna().sum() == 0:
    st.warning(f"Para o tipo `{tipo_escolhido}`, não há valores numéricos válidos em `valor`.")
else:
    # Define top N municípios por média de valor no período
    rank_media = df_tipo.groupby("Município")["valor"].mean().sort_values(ascending=False)
    if limitar_municipios:
        top_muns = rank_media.head(top_n).index
        df_plot_tipo = df_tipo[df_tipo["Município"].isin(top_muns)]
    else:
        df_plot_tipo = df_tipo

    # Gráfico
    plot_serie_temporal(
        df_plot_tipo,
        x_col="ano",
        y_col="valor",
        group_col="Município",
        xlabel="Ano",
        ylabel="Valor do indicador",
        title=f"Série temporal – tipo: {tipo_escolhido}",
    )

    # ====== TEXTO DE INTERPRETAÇÃO (SEÇÃO 1) ======
    st.subheader("Interpretação – série por tipo")

    anos_min, anos_max = min(anos_disponiveis), max(anos_disponiveis)
    top3 = rank_media.head(3)

    descricao_top = ""
    if len(top3) >= 1:
        descricao_top += f"- **{top3.index[0]}** apresenta a maior média no indicador `{tipo_escolhido}` no período, com valor aproximado de **{top3.iloc[0]:.2f}**.\n"
    if len(top3) >= 2:
        descricao_top += f"- **{top3.index[1]}** aparece em segundo lugar, com média em torno de **{top3.iloc[1]:.2f}**.\n"
    if len(top3) >= 3:
        descricao_top += f"- **{top3.index[2]}** ocupa a terceira posição, com média próxima de **{top3.iloc[2]:.2f}**.\n"

    # Tendência global (média por ano)
    media_ano = df_tipo.groupby("ano")["valor"].mean().sort_values()
    if len(media_ano) >= 2:
        primeiro_ano, ultimo_ano = media_ano.index[0], media_ano.index[-1]
        valor_ini, valor_fim = media_ano.iloc[0], media_ano.iloc[-1]

        if valor_ini > valor_fim * 1.05:
            tendencia = (
                f"Observa-se uma **tendência geral de crescimento** do indicador `{tipo_escolhido}` "
                f"ao longo dos anos, saindo de aproximadamente **{valor_fim:.2f}** em {primeiro_ano} "
                f"para cerca de **{valor_ini:.2f}** em {ultimo_ano}."
            )
        elif valor_ini < valor_fim * 0.95:
            tendencia = (
                f"Há indícios de uma **tendência geral de queda** no indicador `{tipo_escolhido}`, "
                f"com valores médios passando de cerca de **{valor_fim:.2f}** em {primeiro_ano} "
                f"para aproximadamente **{valor_ini:.2f}** em {ultimo_ano}."
            )
        else:
            tendencia = (
                f"O indicador `{tipo_escolhido}` apresenta uma **variação relativamente estável** "
                f"ao longo do período, com média em torno de **{valor_ini:.2f}–{valor_ini:.2f}**."
            )
    else:
        tendencia = "Não foi possível avaliar a tendência temporal por falta de dados suficientes."

    st.markdown(f"""
**O que este gráfico responde:**

- Como o indicador **`{tipo_escolhido}`** evoluiu nos municípios entre **{anos_min}** e **{anos_max}**.
- Quais municípios se destacam com **maiores valores médios**.
- Se há uma **tendência geral** de crescimento, queda ou estabilidade no estado.

**Principais destaques (média no período {anos_min}–{anos_max}):**

{descricao_top if descricao_top else "- Não foi possível calcular o ranking – verifique os dados."}

**Tendência global do indicador `{tipo_escolhido}`:**

- {tendencia}
""")

# ===================================================================
# SEÇÃO 2 – SÉRIE POR MUNICÍPIO (VÁRIOS TIPOS) COM NORMALIZAÇÃO
# ===================================================================
st.markdown("---")
st.header("Série temporal por município — comparação de tipos (com normalização)")

st.markdown("""
Aqui fixamos um **município** e comparamos **diferentes indicadores** (tipos)
ao longo do tempo.

Para evitar que indicadores em escala muito diferente (por exemplo, total de vacinas
versus número de hospitais) fiquem incomparáveis, você pode escolher uma **escala normalizada**.

- **Valor original**: números reais (boa para comparar indicadores na mesma ordem de grandeza);
- **Min–Max por tipo (0–1)**: comprime cada indicador para o intervalo `[0, 1]`;
- **Z-score por tipo**: centraliza em 0 com desvio padrão 1 (bom para ver desvios relativos).
""")

col_mun, col_tipos, col_norm = st.columns([2, 2, 2])

with col_mun:
    # tenta defaultar Florianópolis se existir
    default_mun_index = 0
    for cand in ["FLORIANOPOLIS", "Florianópolis", "FLORIANÓPOLIS"]:
        if cand in municipios_disponiveis:
            default_mun_index = municipios_disponiveis.index(cand)
            break

    municipio_escolhido = st.selectbox(
        "Escolha o município:",
        options=municipios_disponiveis,
        index=default_mun_index
    )

with col_tipos:
    # tenta defaultar para ["geral", "covid"] se existirem
    defaults = [t for t in ["geral", "covid"] if t in tipos_disponiveis]
    if not defaults:
        defaults = tipos_disponiveis[:2]

    tipos_selecionados = st.multiselect(
        "Escolha os tipos a comparar no município:",
        options=tipos_disponiveis,
        default=defaults
    )

with col_norm:
    escala_sec2 = st.selectbox(
        "Escala para o gráfico:",
        options=["Valor original", "Min–Max por tipo (0–1)", "Z-score por tipo"],
        index=1,  # default min-max
        key="escala_sec2"
    )


col_valor_sec2 = "valor"
ylabel_sec2 = "Valor do indicador"
if escala_sec2 == "Min–Max por tipo (0–1)":
    col_valor_sec2 = "valor_minmax_tipo"
    ylabel_sec2 = "Indicador normalizado (min–max por tipo)"
elif escala_sec2 == "Z-score por tipo":
    col_valor_sec2 = "valor_zscore_tipo"
    ylabel_sec2 = "Indicador normalizado (z-score por tipo)"

df_city = df_long[
    (df_long["Município"] == municipio_escolhido) &
    (df_long["tipo"].isin(tipos_selecionados))
].copy()

if df_city.empty or df_city[col_valor_sec2].notna().sum() == 0:
    st.warning("Não há dados suficientes para essa combinação de município, tipos e escala.")
else:
    plot_serie_temporal(
        df_city,
        x_col="ano",
        y_col=col_valor_sec2,
        group_col="tipo",
        xlabel="Ano",
        ylabel=ylabel_sec2,
        title=f"Séries temporais por tipo – {municipio_escolhido} ({escala_sec2})",
    )

    # ====== TEXTO DE INTERPRETAÇÃO (SEÇÃO 2) ======
    st.subheader("Interpretação – comparação dentro do município")

    anos_min_city, anos_max_city = df_city["ano"].min(), df_city["ano"].max()
    ano_final = df_city["ano"].max()
    df_final_ano = df_city[df_city["ano"] == ano_final]

    resumo_final = df_final_ano.groupby("tipo")[col_valor_sec2].mean().sort_values(ascending=False)

    texto_resumo = ""
    if not resumo_final.empty:
        # Top indicador no último ano
        tipo_top = resumo_final.index[0]
        valor_top = resumo_final.iloc[0]
        texto_resumo += (
            f"- No ano mais recente (**{ano_final}**), o indicador **`{tipo_top}`** "
            f"apresenta o maior valor médio na escala escolhida (**{escala_sec2}**), com cerca de **{valor_top:.2f}**.\n"
        )

        if len(resumo_final) > 1:
            tipo_low = resumo_final.index[-1]
            valor_low = resumo_final.iloc[-1]
            diff = valor_top - valor_low
            texto_resumo += (
                f"- O indicador com menor valor no mesmo ano é **`{tipo_low}`**, "
                f"com média de aproximadamente **{valor_low:.2f}**, resultando em "
                f"uma diferença de cerca de **{diff:.2f}** na escala escolhida.\n"
            )

        texto_resumo += (
            "- A normalização permite observar o **comportamento relativo** dos indicadores, "
            "independentemente da unidade original (por exemplo, número de vacinas, leitos ou centros).\n"
        )
    else:
        texto_resumo += "- Não foi possível calcular o resumo para o ano final.\n"

    st.markdown(f"""
**O que este gráfico responde para `{municipio_escolhido}`:**

- Como cada indicador evolui entre **{anos_min_city}** e **{anos_max_city}**.
- Qual indicador domina (valores mais altos) no ano mais recente, **depois de normalizar**.
- Se há divergências importantes na trajetória dos indicadores ao longo do tempo.

**Resumo automático para o ano mais recente ({ano_final}) na escala `{escala_sec2}`:**

{texto_resumo}
""")

# ===================================================================
# SEÇÃO 3 – COMPARAÇÃO POR ANO (RANKING DE MUNICÍPIOS)
# ===================================================================
st.markdown("---")
st.header("Comparação por ano — ranking de municípios (valor original)")

st.markdown("""
Aqui fixamos um **ano** e um **tipo de indicador**, e mostramos um
**ranking de municípios** usando os valores originais.

Bom para responder:

- Em determinado ano, quais municípios lideram o indicador?
- Qual é a diferença entre o melhor e o pior colocado?
""")

col_tipo_ano1, col_tipo_ano2 = st.columns(2)

with col_tipo_ano1:
    tipo_rank = st.selectbox(
        "Tipo de indicador para o ranking:",
        options=tipos_disponiveis,
        index=default_tipo
    )

with col_tipo_ano2:
    ano_rank = st.selectbox(
        "Ano para o ranking:",
        options=anos_disponiveis,
        index=len(anos_disponiveis) - 1  # último ano
    )

df_rank = df_long[
    (df_long["tipo"] == tipo_rank) &
    (df_long["ano"] == ano_rank)
].copy()

if df_rank.empty or df_rank["valor"].notna().sum() == 0:
    st.warning("Não há dados numéricos para esse ano e tipo.")
else:
    df_rank_sorted = df_rank.sort_values("valor", ascending=False)

    fig_rank, ax_rank = plt.subplots(figsize=(10, 6))
    ax_rank.barh(df_rank_sorted["Município"], df_rank_sorted["valor"])
    ax_rank.invert_yaxis()
    ax_rank.set_title(f"Ranking de municípios – tipo `{tipo_rank}` em {ano_rank}")
    ax_rank.set_xlabel("Valor do indicador (original)")
    ax_rank.set_ylabel("Município")
    fig_rank.tight_layout()
    st.pyplot(fig_rank)

    # ====== TEXTO DE INTERPRETAÇÃO (SEÇÃO 3) ======
    st.subheader("Interpretação – ranking por ano")

    top3_rank = df_rank_sorted.head(3)
    bottom3_rank = df_rank_sorted.tail(3)

    texto_rank = ""

    if not top3_rank.empty:
        texto_rank += f"- O município com maior valor no indicador **`{tipo_rank}`** em **{ano_rank}** é **{top3_rank.iloc[0]['Município']}**, com aproximadamente **{top3_rank.iloc[0]['valor']:.2f}**.\n"
    if len(top3_rank) >= 2:
        texto_rank += f"- Em seguida aparecem **{top3_rank.iloc[1]['Município']}** (~{top3_rank.iloc[1]['valor']:.2f}) e **{top3_rank.iloc[2]['Município']}** (~{top3_rank.iloc[2]['valor']:.2f}).\n"

    if len(df_rank_sorted) > 3:
        texto_rank += (
            f"- Na outra extremidade, os menores valores são observados em municípios como "
            f"**{bottom3_rank.iloc[0]['Município']}**, **{bottom3_rank.iloc[1]['Município']}** "
            f"e **{bottom3_rank.iloc[2]['Município']}**.\n"
        )

    max_val = df_rank_sorted["valor"].max()
    min_val = df_rank_sorted["valor"].min()
    diff_val = max_val - min_val

    texto_rank += (
        f"- A diferença entre o maior e o menor valor do indicador em {ano_rank} é de aproximadamente **{diff_val:.2f}**, "
        "sugerindo o grau de desigualdade entre municípios nesse indicador.\n"
    )

    st.markdown(f"""
**O que este gráfico responde:**

- Quais municípios **lideram** o indicador `{tipo_rank}` no ano **{ano_rank}`.
- Quais municípios apresentam os **menores valores**.
- Qual é a **amplitude** entre o melhor e o pior desempenho.

**Resumo automático:**

{texto_rank}
""")

# ===================================================================
# SEÇÃO 4 – COMPARAR VÁRIAS CIDADES E TIPOS (NORMALIZADO)
# ===================================================================
st.markdown("---")
st.header("Comparação entre cidades e tipos — com normalização")

st.markdown("""
Nesta seção você pode:

- Escolher **um ou mais tipos de indicador** (por exemplo, `covid`, `geral`, `centros`, `leitos`, ...);
- Escolher **um conjunto de municípios**;
- Escolher a **escala** (original, min–max ou z-score).

O gráfico mostra uma série temporal para cada combinação **Município–Tipo**,
permitindo comparar, por exemplo:

- Florianópolis (covid) vs Joinville (covid);
- Florianópolis (covid, geral) vs Blumenau (covid, geral);
- etc.
""")

col_tipos4, col_cidades4, col_norm4 = st.columns([2, 2, 2])

with col_tipos4:
    tipos_sel4 = st.multiselect(
        "Tipos de indicador:",
        options=tipos_disponiveis,
        default=[t for t in ["covid", "geral"] if t in tipos_disponiveis] or tipos_disponiveis[:2]
    )

with col_cidades4:
    # Default: algumas cidades “grandes” se existirem, senão primeiras da lista
    defaults_cidades = [c for c in ["FLORIANOPOLIS", "JOINVILLE", "BLUMENAU"] if c in municipios_disponiveis]
    if not defaults_cidades:
        defaults_cidades = municipios_disponiveis[:3]

    cidades_sel4 = st.multiselect(
        "Municípios a comparar:",
        options=municipios_disponiveis,
        default=defaults_cidades
    )

with col_norm4:
    escala_sec4 = st.selectbox(
        "Escala para o gráfico:",
        options=["Valor original", "Min–Max por tipo (0–1)", "Z-score por tipo"],
        index=1,
        key="escala_sec4"
    )


col_valor_sec4 = "valor"
ylabel_sec4 = "Valor do indicador"
if escala_sec4 == "Min–Max por tipo (0–1)":
    col_valor_sec4 = "valor_minmax_tipo"
    ylabel_sec4 = "Indicador normalizado (min–max por tipo)"
elif escala_sec4 == "Z-score por tipo":
    col_valor_sec4 = "valor_zscore_tipo"
    ylabel_sec4 = "Indicador normalizado (z-score por tipo)"

df_comp = df_long[
    (df_long["tipo"].isin(tipos_sel4)) &
    (df_long["Município"].isin(cidades_sel4))
].copy()

if df_comp.empty or df_comp[col_valor_sec4].notna().sum() == 0:
    st.warning("Não há dados suficientes para essa combinação de cidades, tipos e escala.")
else:
    # Cria uma label combinando Município e Tipo
    df_comp["serie"] = df_comp["Município"] + " – " + df_comp["tipo"]

    plot_serie_temporal(
        df_comp,
        x_col="ano",
        y_col=col_valor_sec4,
        group_col="serie",
        xlabel="Ano",
        ylabel=ylabel_sec4,
        title=f"Comparação entre cidades e tipos ({escala_sec4})",
    )

    st.subheader("Interpretação – comparação entre cidades e tipos")

    anos_min4, anos_max4 = df_comp["ano"].min(), df_comp["ano"].max()
    ano_final4 = df_comp["ano"].max()
    df_final4 = df_comp[df_comp["ano"] == ano_final4]

    resumo4 = df_final4.groupby("serie")[col_valor_sec4].mean().sort_values(ascending=False)

    texto4 = ""
    if not resumo4.empty:
        serie_top = resumo4.index[0]
        valor_top = resumo4.iloc[0]
        texto4 += (
            f"- No ano mais recente (**{ano_final4}**), a combinação **{serie_top}** "
            f"apresenta o maior valor médio na escala `{escala_sec4}`, com cerca de **{valor_top:.2f}**.\n"
        )

        if len(resumo4) > 1:
            serie_low = resumo4.index[-1]
            valor_low = resumo4.iloc[-1]
            diff4 = valor_top - valor_low
            texto4 += (
                f"- A combinação com menor valor é **{serie_low}**, com média de aproximadamente **{valor_low:.2f}**, "
                f"resultando em uma diferença de cerca de **{diff4:.2f}** entre o maior e o menor caso.\n"
            )

        texto4 += (
            "- A normalização por tipo permite comparar diretamente diferentes indicadores e cidades, "
            "destacando quais contextos estão mais acima ou abaixo da média relativa.\n"
        )
    else:
        texto4 += "- Não foi possível calcular um resumo adequado para o ano final.\n"

    st.markdown(f"""
**O que este gráfico responde:**

- Como diferentes **cidades** se comparam em um mesmo tipo de indicador;
- Como diferentes **tipos** se comportam em mais de uma cidade;
- Quais combinações cidade–tipo se destacam positivamente ou negativamente.

**Resumo automático para o ano mais recente ({ano_final4}) na escala `{escala_sec4}`:**

{texto4}
""")
