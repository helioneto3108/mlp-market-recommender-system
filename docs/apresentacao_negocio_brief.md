# Brief para gerar a apresentação (PPTX) — colar no Claude (web)

> **Como usar:** copie este arquivo inteiro e cole no terminal do Claude (web).
> Ele contém o *prompt* de instrução + todo o conteúdo dos slides. O agente deve
> gerar o arquivo `docs/apresentacao_negocio.pptx`.

---

## PROMPT PARA O AGENTE (instruções)

Gere uma apresentação em **PowerPoint** (`docs/apresentacao_negocio.pptx`) usando
**python-pptx** (instale com `uv add --dev python-pptx` ou `pip install python-pptx`).

**Público:** analistas de negócio, com pouco/médio conhecimento de algoritmos e
dados. **Idioma:** português (PT-BR). **Tom:** claro, direto, sem jargão; quando
usar um termo técnico, explique em uma frase com analogia.

**Diretrizes de design:**
- ~13 slides (1 título + conteúdo + encerramento). Um slide = uma ideia.
- Fonte legível (ex.: Calibri/Arial), títulos grandes, poucos bullets por slide.
- Paleta consistente por modelo (use sempre as mesmas cores quando citar modelos):
  Popularidade = cinza, Histórico = azul, IA simples (NCF) = laranja,
  **IA com contexto (MLP) = verde** (o vencedor, destacado).
- Inclua as duas imagens do repositório como slides/figuras:
  - `docs/img/model/01_model_comparison.png` (comparação dos modelos)
  - `docs/img/features/01_recency_vs_reorder.png` (recência prevê recompra)
- Use as **notas do apresentador** (speaker notes) de cada slide — o conteúdo
  marcado como "🎤 Notas" abaixo deve ir nas notas, não no slide.
- Evite tabelas densas; prefira números grandes em destaque e 1 gráfico por slide.

Conteúdo exato de cada slide a seguir.

---

## CONTEÚDO DOS SLIDES

### Slide 1 — Capa
**Título:** Recomendação Inteligente de Produtos
**Subtítulo:** Como prever o próximo pedido de cada cliente — e por que isso vende mais
**Rodapé:** Tech Challenge • Projeto de Ciência de Dados

🎤 Notas: Apresentação dos resultados de um sistema que aprende o comportamento de
compra dos clientes para recomendar produtos. Foco no valor de negócio.

---

### Slide 2 — O problema de negócio
**Título:** O desafio
- Um e-commerce de supermercado quer **recomendar os produtos certos** para cada cliente.
- Recomendação boa = mais **conversão**, mais **recompra** e melhor **experiência**.
- A pergunta central: *"Quais produtos este cliente vai querer no próximo pedido?"*

🎤 Notas: Em supermercado, grande parte das compras é **recompra** (itens que a
pessoa já consome). Acertar a recomendação reduz fricção e aumenta o ticket.

---

### Slide 3 — Os dados que temos
**Título:** O que sabemos sobre os clientes
- **~206 mil clientes** e **mais de 3 milhões de pedidos** reais (base pública Instacart).
- **~50 mil produtos** diferentes.
- Para cada cliente, o **histórico de compras**: o que comprou, quantas vezes e quando.
- Não temos "nota" do cliente — temos o comportamento real: **comprou / recomprou**.

🎤 Notas: É "feedback implícito": em vez de estrelas/avaliações, usamos a ação de
comprar como sinal de interesse. Cada cliente tem entre 4 e 100 pedidos de histórico.

---

### Slide 4 — Como medimos "acerto" (a régua)
**Título:** Como sabemos se a recomendação é boa?
- Para cada cliente, escondemos o **último pedido real** (o que ele de fato comprou).
- O sistema gera uma **lista de 10 recomendações**.
- Comparamos: *quantos dos 10 ele realmente comprou?*
- Duas perguntas-chave:
  - **Precisão** — dos 10 que recomendei, quantos ele comprou?
  - **Cobertura** — do que ele comprou, quanto eu cobri com os 10?

🎤 Notas: Avaliamos em **26 mil clientes que o modelo nunca viu no treino** — teste
justo, como se fossem clientes novos. Também usamos métricas que premiam acertar
nas **primeiras** posições da lista (o cliente olha o topo).

---

### Slide 5 — As abordagens testadas
**Título:** Do simples ao inteligente — testamos 4 abordagens
- **1. Mais vendidos** (popularidade) — recomenda o campeão de vendas para todos. Não personaliza.
- **2. "O de sempre"** (histórico) — recomenda o que o próprio cliente mais compra.
- **3. IA "ingênua"** — rede neural que só conhece *identidades* (cliente X, produto Y).
- **4. IA com contexto** — rede neural que também enxerga **frequência e recência** de compra. ✅

🎤 Notas: A lógica é ir do trivial ao sofisticado para provar que a IA realmente
agrega valor — e não usar IA "porque é moderno". A nº 2 (histórico) é um
concorrente forte e difícil de superar.

---

### Slide 6 — O resultado principal
**Título:** Qual abordagem recomenda melhor?
**(inserir imagem `docs/img/model/01_model_comparison.png` em destaque)**
- A **IA com contexto (verde)** vence em **todas** as métricas.
- Supera "o de sempre" (histórico) e fica muito acima de "mais vendidos".

🎤 Notas: Quanto mais alto, melhor. As barras verdes (IA com contexto) lideram em
todas as quatro métricas. A IA "ingênua" (laranja) empata com "mais vendidos" —
detalhe importante explicado no próximo slide.

---

### Slide 7 — Tradução para o negócio
**Título:** O que esses números significam na prática
- Com a **IA com contexto**, a cada **10 produtos recomendados**:
  - **~3 o cliente realmente compra** no próximo pedido (precisão ~30%).
  - As 10 sugestões **cobrem ~35%** de toda a próxima cesta dele.
- **Qualidade do ranking ~10% superior** à melhor regra simples ("o de sempre").
- E **~80% superior** à abordagem de "mais vendidos".

🎤 Notas: 3 em 10 acertos personalizados é um resultado forte para recomendação de
catálogo amplo. Os ~10% e ~80% referem-se à métrica NDCG (qualidade da ordenação:
0,43 vs 0,39 vs 0,24).

---

### Slide 8 — A "virada de chave": recência
**Título:** Por que a IA com contexto ganha
**(inserir imagem `docs/img/features/01_recency_vs_reorder.png`)**
- O sinal mais forte: **há quanto tempo** o cliente comprou o item.
- Comprou **recentemente** → altíssima chance de comprar de novo.
- A IA "ingênua" não enxerga isso; a IA com contexto, sim.

🎤 Notas: O gráfico mostra que a probabilidade de recompra **despenca** conforme o
item fica "esquecido". A IA com contexto também entende **similaridade entre
produtos** (mesma seção/categoria), ajudando inclusive itens menos frequentes.

---

### Slide 9 — A lição técnica (sem jargão)
**Título:** Não é "IA vs regra" — é informação rica vs informação pobre
- A IA "ingênua" só sabia **quem** é o cliente e **qual** o produto.
- A IA vencedora sabe também **com que frequência** e **há quanto tempo** ele compra.
- Mesma tecnologia, **informação melhor** → resultado melhor.

🎤 Notas: Mensagem-chave para o público: o ganho não veio de "mais IA", e sim de
**alimentar o modelo com o contexto certo**. Investir em entender o dado vale tanto
quanto o algoritmo.

---

### Slide 10 — Valor para o negócio
**Título:** O que isso destrava
- **Personalização real** — cada cliente vê o que tende a querer.
- Potencial de **mais conversão e recompra** e **ticket médio** maior.
- Base para **campanhas, lembretes de recompra e "compre de novo"**.

🎤 Notas: Os ganhos de métrica se traduzem em recomendações mais relevantes na
vitrine, no carrinho e em e-mails — alavancas diretas de receita.

---

### Slide 11 — Limitações (transparência)
**Título:** O que o modelo ainda não faz
- Recomenda muito bem o que o cliente **já consome**; é mais limitado para itens
  **100% novos** (que ele nunca comprou).
- Baseia-se no **comportamento passado** — clientes muito novos têm menos sinal.

🎤 Notas: Honestidade gera confiança. O próximo ciclo pode incluir descoberta de
novidades (cross-sell) e enriquecer o perfil de clientes novos.

---

### Slide 12 — Próximos passos
**Título:** Do protótipo ao produto
- **Rastreabilidade** — registrar versões e métricas dos modelos (governança).
- **Empacotamento** — preparar para rodar de forma confiável e reproduzível.
- **Publicação** — disponibilizar as recomendações para os sistemas do negócio.

🎤 Notas: Em termos técnicos: MLflow (registro/governança de modelos), Docker
(empacotamento) e pipeline de dados versionado. Tradução: tornar o modelo
confiável, auditável e pronto para uso.

---

### Slide 13 — Encerramento
**Título:** Resumo
- Construímos um recomendador que **supera as regras simples** com folga.
- O segredo foi **dar contexto certo** (frequência e recência) à IA.
- Pronto para evoluir rumo à **produção** e gerar valor de negócio.
**Rodapé:** Obrigado! • Perguntas?

🎤 Notas: Fechar reforçando a mensagem: dado bem trabalhado + IA = recomendação que
vende. Abrir para perguntas.

---

## DADOS DE APOIO (caso o agente precise conferir números)

| Modelo | Precisão@10 | Cobertura@10 | Qualidade do ranking (NDCG@10) |
|---|---|---|---|
| Mais vendidos (popularidade) | 0,169 | 0,214 | 0,237 |
| O de sempre (histórico) | 0,267 | 0,320 | 0,388 |
| IA ingênua (NCF, só IDs) | 0,176 | 0,222 | 0,243 |
| **IA com contexto (MLP)** | **0,298** | **0,353** | **0,428** |

- Avaliação: 26.241 clientes "held-out" (não vistos no treino), top-10 vs próximo pedido real.
- Ganho da IA com contexto sobre o histórico: **+10%** no NDCG; sobre popularidade: **+80%**.
- Base: Instacart — ~206 mil clientes, +3 milhões de pedidos, ~50 mil produtos.
