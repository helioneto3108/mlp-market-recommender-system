# Market Recommender Frontend

Front-end em React + Vite para simular um site dark de recomendações de compras com base em perfis de usuários.

## O que vem pronto

- Tela dark estilo marketplace + dashboard de IA
- 3 perfis simulados: Usuário A, B e C
- Top 10 recomendações por perfil
- Cards com ícones/desenhos, sem imagens reais
- Barras de score em porcentagem
- Botão para simular execução do modelo
- Painel com métricas do modelo `mlp_temporal_v1`
- Estrutura preparada para trocar mock por API futuramente

## Como rodar

```bash
npm install
npm run dev
```

Depois abra o endereço mostrado no terminal, normalmente:

```bash
http://localhost:5173
```

## Onde alterar os dados

Os perfis e recomendações estão em:

```txt
src/data/mockRecommendations.js
```

Para conectar com uma API depois, substitua o mock por uma chamada parecida com:

```js
const response = await fetch(`http://localhost:8000/recommendations/${profileId}`);
const data = await response.json();
```

## Sugestão de retorno da API

```json
{
  "profile_id": "A",
  "top_k": 10,
  "recommendations": [
    {
      "rank": 1,
      "product_id": 13176,
      "product_name": "Bag of Organic Bananas",
      "category": "Fresh Produce",
      "aisle_id": 24,
      "department_id": 4,
      "score": 2.84,
      "score_percent": 96
    }
  ]
}
```
