# Guia de Personalização - Market Recommender

Este guia mostra onde alterar o front para funcionar com o usuário 10 e com a API local.

## 1. Configuração da API

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure por:

```js
export const API_USER_URL = "http://127.0.0.1:8010/users/10";
```

Se a rota mudar, altere apenas essa linha.

Exemplos:

```js
export const API_USER_URL = "http://127.0.0.1:8010/users/25";
```

ou:

```js
export const API_USER_URL = "http://localhost:8010/recommendations/10";
```

Importante: use `http://` antes do endereço. No navegador, `fetch("127.0.0.1:8010/users/10")` pode falhar.

---

## 2. Formato esperado da resposta

O front espera receber da API:

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
    },
    {
      "product_id": 48720,
      "product_name": "Shoestring Fries",
      "user_product_purchase_count": 0,
      "user_product_reorder_rate": 0.0
    }
  ]
}
```

---

## 3. Onde alterar produtos locais

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure por:

```js
export const fallbackApiUser = { ... }
```

Esses dados aparecem quando a API está desligada ou com erro.

---

## 4. Onde alterar os ícones dos produtos

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure pela função:

```js
function getProductIcon(productName = "")
```

Ela define o ícone de acordo com o nome do produto.

Exemplo:

```js
if (name.includes("fries") || name.includes("potato")) return "🍟";
```

Para adicionar outro produto:

```js
if (name.includes("milk")) return "🥛";
```

---

## 5. Onde alterar categorias dos produtos

Arquivo:

```txt
src/data/mockRecommendations.js
```

Procure pela função:

```js
function getProductCategory(productName = "")
```

Exemplo:

```js
if (name.includes("chorizo") || name.includes("pork")) return "Meat / Sausage";
```

---

## 6. Onde alterar textos da tela

Arquivo:

```txt
src/App.jsx
```

Nesse arquivo ficam:

- título do site
- subtítulo
- nome dos blocos
- botão `Refresh from API`
- cards principais
- tabela de produtos

---

## 7. Onde alterar visual, cores e tema dark

Arquivo:

```txt
src/styles/global.css
```

No começo do arquivo existem variáveis principais:

```css
--bg: #070a12;
--green: #44f2a1;
--blue: #66a9ff;
--purple: #b78cff;
--amber: #ffd166;
```

---

## 8. Onde o front chama a API

Arquivo:

```txt
src/App.jsx
```

Procure por:

```js
async function loadUserFromApi() {
```

É essa função que faz:

```js
const response = await fetch(API_USER_URL);
const data = await response.json();
const nextProfile = buildProfileFromApiUser(data);
```

---

## 9. Atenção ao CORS

Se a API está rodando, mas o navegador bloqueia a chamada, habilite CORS no backend.

Exemplo para FastAPI:

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

---

## 10. Observação sobre porcentagens

A rota atual enviada retorna:

```json
"user_product_purchase_count": 0,
"user_product_reorder_rate": 0.0
```

Como ainda não existe `score_percent` na resposta, o front cria uma pontuação visual baseada no ranking dos produtos.

Se a API futuramente retornar um campo assim:

```json
"score_percent": 91
```

ou:

```json
"recommendation_score": 2.31
```

o front já tenta usar esse valor automaticamente.
