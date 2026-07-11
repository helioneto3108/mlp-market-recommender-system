// ============================================================
// CONFIGURAÇÃO PRINCIPAL DO MARKET RECOMMENDER
//
// Este arquivo concentra as informações que seus colegas mais
// provavelmente vão alterar:
// 1. URL da API do usuário
// 2. Métricas do modelo exibidas no painel
// 3. Dados fallback, usados se a API estiver desligada
// 4. Conversão da resposta da API para o formato visual do React
//
// API esperada neste front:
// GET http://127.0.0.1:8010/users/10
//
// Formato esperado da resposta:
// {
//   "user_id": 10,
//   "total_orders": 6,
//   "avg_days_between_orders": 21.8,
//   "avg_order_hour": 17.0,
//   "favorite_order_day": "3",
//   "products": [
//     {
//       "product_id": 29650,
//       "product_name": "Pork Chorizo",
//       "user_product_purchase_count": 0,
//       "user_product_reorder_rate": 0.0
//     }
//   ]
// }
// ============================================================

// Troque esta URL se a porta, rota ou usuário mudarem.
// Importante: no fetch precisa ter http:// antes do endereço.
export const API_USER_URL = "http://127.0.0.1:8010/users/10";

export const TOP_RECOMMENDATIONS_LIMIT = 10;

export const modelStats = {
  name: "mlp_temporal_v1",
  description: "MLP with user, product, aisle and department embeddings",
  primaryMetric: "NDCG@10",
  primaryK: 10,
  embeddingDim: 32,
  hiddenLayers: "128 → 64",
  dropout: 0.25,
  bestEpoch: 2,
  metrics: [
    { label: "Random seed", value: "42" },
    { label: "Batch size", value: "8192" },
    { label: "Learning rate", value: "0.001" },
    { label: "Dropout", value: "0.25" }
  ]
};

// Este objeto é usado como fallback local.
// Se a API estiver desligada, o front continua abrindo com esses dados.
export const fallbackApiUser = {
  user_id: 10,
  total_orders: 6,
  avg_days_between_orders: 21.8,
  avg_order_hour: 17.0,
  favorite_order_day: "3",
  products: [
    {
      product_id: 29650,
      product_name: "Pork Chorizo",
      user_product_purchase_count: 0,
      user_product_reorder_rate: 0.0
    },
    {
      product_id: 48720,
      product_name: "Shoestring Fries",
      user_product_purchase_count: 0,
      user_product_reorder_rate: 0.0
    },
    {
      product_id: 24654,
      product_name: "Potato Hot Dog Buns",
      user_product_purchase_count: 0,
      user_product_reorder_rate: 0.0
    },
    {
      product_id: 10177,
      product_name: "German Barrel Sauerkraut",
      user_product_purchase_count: 0,
      user_product_reorder_rate: 0.0
    }
  ]
};

const orderDayLabels = {
  0: "Day 0",
  1: "Day 1",
  2: "Day 2",
  3: "Day 3",
  4: "Day 4",
  5: "Day 5",
  6: "Day 6"
};

function formatDecimal(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "0.0";
}

function formatHour(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "Unknown";
  return `${Math.round(number)}h`;
}

function getFavoriteDayLabel(value) {
  return orderDayLabels[value] ?? `Day ${value}`;
}

function getProductIcon(productName = "") {
  const name = productName.toLowerCase();

  // Vegetais e ervas
  if (name.includes("cauliflower")) return "🥦";
  if (name.includes("asparagus")) return "🌱";
  if (name.includes("parsley")) return "🌿";
  if (name.includes("oregano")) return "🌿";
  if (name.includes("onion")) return "🧅";
  if (name.includes("garlic")) return "🧄";
  if (name.includes("cabbage") || name.includes("sauerkraut")) return "🥬";
  if (name.includes("lettuce")) return "🥬";
  if (name.includes("carrot")) return "🥕";
  if (name.includes("tomato")) return "🍅";
  if (name.includes("potato")) return "🥔";

  // Frutas
  if (name.includes("cherries") || name.includes("cherry")) return "🍒";
  if (name.includes("banana")) return "🍌";
  if (name.includes("apple")) return "🍎";
  if (name.includes("orange")) return "🍊";
  if (name.includes("berry") || name.includes("strawberry")) return "🍓";
  if (name.includes("avocado")) return "🥑";

  // Carnes e embutidos
  if (name.includes("turkey") || name.includes("bacon")) return "🥓";
  if (name.includes("chicken")) return "🍗";
  if (name.includes("sausage") || name.includes("chorizo")) return "🌭";
  if (name.includes("pork") || name.includes("beef") || name.includes("meat")) return "🥩";

  // Molhos, temperos e pantry
  if (name.includes("mustard") || name.includes("dijon")) return "🟡";
  if (name.includes("powder") || name.includes("spice") || name.includes("seasoning")) return "🧂";

  // Padaria e laticínios
  if (name.includes("bread") || name.includes("bagel") || name.includes("toast")) return "🍞";
  if (name.includes("buns") || name.includes("bun") || name.includes("hot dog")) return "🌭";
  if (name.includes("milk")) return "🥛";
  if (name.includes("cheese")) return "🧀";
  if (name.includes("yogurt")) return "🥣";
  if (name.includes("egg")) return "🥚";

  // Congelados e snacks
  if (name.includes("fries") || name.includes("french fry")) return "🍟";
  if (name.includes("pizza")) return "🍕";
  if (name.includes("cookie")) return "🍪";
  if (name.includes("chocolate")) return "🍫";
  if (name.includes("ice cream")) return "🍨";
  if (name.includes("cereal")) return "🥣";
  if (name.includes("pasta")) return "🍝";
  if (name.includes("rice")) return "🍚";

  // Bebidas
  if (name.includes("coffee")) return "☕";
  if (name.includes("tea")) return "🍵";
  if (name.includes("water")) return "💧";
  if (name.includes("juice")) return "🧃";

  return "🛒";
}

function getProductCategory(productName = "") {
  const name = productName.toLowerCase();

  if (name.includes("fries") || name.includes("potato")) return "Frozen / Potato";
  if (name.includes("hot dog") || name.includes("buns")) return "Bakery / Buns";
  if (name.includes("chorizo") || name.includes("pork")) return "Meat / Sausage";
  if (name.includes("sauerkraut") || name.includes("cabbage")) return "Pickled / Pantry";

  return "Market product";
}

function getBadge(product, index) {
  const reorderRate = Number(product.user_product_reorder_rate ?? 0);
  const purchaseCount = Number(product.user_product_purchase_count ?? 0);

  if (index === 0) return "Top Candidate";
  if (reorderRate > 0.5) return "Reorder Signal";
  if (purchaseCount > 0) return "Known Product";
  return "Model Pick";
}

function getDisplayScore(product, index) {
  // Se a API futuramente retornar score_percent, usamos o valor real.
  if (product.score_percent !== undefined) return Math.round(Number(product.score_percent));

  // Se retornar score ou recommendation_score, convertemos para visual.
  const rawScore = product.recommendation_score ?? product.score;
  if (rawScore !== undefined && Number.isFinite(Number(rawScore))) {
    const normalized = 55 + Math.tanh(Number(rawScore)) * 40;
    return Math.max(1, Math.min(99, Math.round(normalized)));
  }

  // Como a sua rota atual retorna purchase_count/reorder_rate zerados,
  // criamos uma pontuação visual baseada no ranking para apresentação.
  return Math.max(62, 94 - index * 7);
}

function normalizeProduct(product, index) {
  const productName = product.product_name ?? product.name ?? `Product ${product.product_id}`;

  return {
    rank: index + 1,
    productId: product.product_id ?? product.productId,
    name: productName,
    category: product.category ?? getProductCategory(productName),
    scorePercent: getDisplayScore(product, index),
    score: product.recommendation_score ?? product.score ?? product.user_product_reorder_rate ?? 0,
    icon: product.icon ?? getProductIcon(productName),
    badge: product.badge ?? getBadge(product, index),
    purchaseCount: product.user_product_purchase_count ?? 0,
    reorderRate: product.user_product_reorder_rate ?? 0,
    aisleId: product.aisle_id,
    departmentId: product.department_id
  };
}

export function buildProfileFromApiUser(apiUser = fallbackApiUser) {
  const products = Array.isArray(apiUser.products) ? apiUser.products : [];
  const recommendations = products
  .slice(0, TOP_RECOMMENDATIONS_LIMIT)
  .map(normalizeProduct);
  const avgDays = formatDecimal(apiUser.avg_days_between_orders, 1);
  const avgHour = formatHour(apiUser.avg_order_hour);
  const favoriteDay = getFavoriteDayLabel(apiUser.favorite_order_day);

  return {
    id: String(apiUser.user_id ?? 10),
    label: `User ${apiUser.user_id ?? 10}`,
    title: "Single Customer Profile",
    icon: "🧺",
    color: "green",
    shortDescription:
      `Customer profile loaded from the API. This user has ${apiUser.total_orders ?? 0} orders, ` +
      `usually buys around ${avgHour}, and has an average gap of ${avgDays} days between orders.`,
    behavior: [
      `${apiUser.total_orders ?? 0} total orders`,
      `${avgDays} days between orders`,
      `favorite order day: ${favoriteDay}`,
      `avg order hour: ${avgHour}`
    ],
    stats: {
      totalOrders: `${apiUser.total_orders ?? 0}`,
      avgDaysBetweenOrders: `${avgDays} days`,
      avgOrderHour: avgHour,
      favoriteOrderDay: favoriteDay
    },
    recommendations
  };
}

export const fallbackProfile = buildProfileFromApiUser(fallbackApiUser);
