# Market Recommender Front-end

Front-end em **React + Vite** com visual dark para simular e visualizar recomendações de um modelo de ML para mercado.

Esta versão foi adaptada para **um único usuário** e busca os dados na API local:

```txt
http://127.0.0.1:8010/users/10
```

## Como rodar

Dentro da pasta do front-end:

```bash
npm install
npm run dev
```

Depois abra o link que aparecer no terminal, normalmente:

```txt
http://localhost:5173
```

## Arquivo principal para configurar

As principais configurações estão em:

```txt
src/data/mockRecommendations.js
```

Nesse arquivo você encontra:

- `API_USER_URL`: URL da API.
- `modelStats`: informações do modelo exibidas na tela.
- `fallbackApiUser`: dados locais usados caso a API esteja desligada.
- `buildProfileFromApiUser`: função que transforma a resposta da API no formato visual usado pelo React.

## Formato esperado da API

O front espera que o endpoint retorne algo nesse formato:

```json
{
  "user_id": 10,
  "total_orders": 6,
  "avg_days_between_orders": 21.8,
  "avg_order_hour": 17.0,
  "favorite_order_day": "3",
  "products": [
    {
      "product_id": 29650,
      "product_name": "Pork Chorizo",
      "user_product_purchase_count": 0,
      "user_product_reorder_rate": 0.0
    }
  ]
}
```

## Onde mexer para personalizar

### Alterar usuário ou rota da API

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure por:

```js
export const API_USER_URL = "http://127.0.0.1:8010/users/10";
```

### Alterar produtos fallback

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure por:

```js
export const fallbackApiUser = { ... }
```

### Alterar textos e layout da tela

Arquivo:

```txt
src/App.jsx
```

### Alterar cores e estilo dark

Arquivo:

```txt
src/styles/global.css
```

## Observação sobre CORS

Se a API estiver rodando, mas o front mostrar erro de conexão, pode ser CORS.
No FastAPI, habilite algo parecido com:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
