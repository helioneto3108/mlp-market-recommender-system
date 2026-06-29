export const PRODUCTS = [
  { id: 24852, emoji: '🍌', name: 'Banana', price: 4.99 },
  { id: 13176, emoji: '🍌', name: 'Bag of Organic Bananas', price: 7.9 },
  { id: 21137, emoji: '🍓', name: 'Organic Strawberries', price: 12.5 },
  { id: 27845, emoji: '🥛', name: 'Organic Whole Milk', price: 8.9 },
  { id: 47209, emoji: '🥑', name: 'Organic Hass Avocado', price: 6.9 },
  { id: 47766, emoji: '🥑', name: 'Organic Avocado', price: 5.9 },
  { id: 21903, emoji: '🥬', name: 'Organic Baby Spinach', price: 9.9 },
  { id: 16797, emoji: '🍓', name: 'Strawberries', price: 10.9 },
  { id: 26209, emoji: '🍋', name: 'Limes', price: 4.5 },
  { id: 47626, emoji: '🍋', name: 'Large Lemon', price: 3.9 },
  { id: 27966, emoji: '🫐', name: 'Organic Raspberries', price: 13.9 },
  { id: 39275, emoji: '🫐', name: 'Organic Blueberries', price: 14.9 },
  { id: 22035, emoji: '🧀', name: 'Organic Whole String Cheese', price: 15.9 },
  { id: 46667, emoji: '🥒', name: 'Organic Cucumber', price: 4.2 },
  { id: 22935, emoji: '🧅', name: 'Organic Yellow Onion', price: 4.9 },
  { id: 45007, emoji: '🥒', name: 'Organic Zucchini', price: 5.9 },
]

export async function fetchRecommendations(cartIds) {
  const cartItems = cartIds
    .map((id) => PRODUCTS.find((product) => product.id === id))
    .filter(Boolean)
    .map((product) => product.name)

  console.log('Produtos enviados para API:', cartItems)

  const response = await fetch('http://127.0.0.1:8000/recommend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      cart_items: cartItems,
      top_n: 5,
      strategy: 'hybrid',
    }),
  })

  if (!response.ok) {
    throw new Error(`Erro na API: ${response.status}`)
  }

  const data = await response.json()

  console.log('Resposta da API:', data)

  return data.recommendations.map((recommendation) => {
    const product = PRODUCTS.find(
      (item) => item.name === recommendation.recommended_product
    )

    return {
      id: product?.id ?? recommendation.recommended_product,
      name: recommendation.recommended_product,
      emoji: product?.emoji ?? '🛒',
      p: recommendation.recommendation_percent,
      finalScore: recommendation.final_score,
      basketScore: recommendation.basket_score_norm,
      similarityScore: recommendation.similarity_score_norm,
    }
  })
}

export function formatPrice(value) {
  return 'R$' + value.toFixed(2).replace('.', ',')
}

export function generateBarcodeBars(seed) {
  const widths = [
    2, 1, 3, 1, 2, 1, 4, 1, 2, 3, 1, 2, 1,
    3, 2, 1, 4, 1, 2, 1, 3, 2, 1, 2, 1, 3,
  ]

  const bars = []
  let x = 0

  widths.forEach((w, i) => {
    const h = seed > 0 ? 30 - ((i * seed * 7) % 8) : 30

    bars.push({
      x,
      w,
      h,
      y: 30 - h,
    })

    x += w + (i % 3 === 0 ? 2 : 1)
  })

  return bars
}