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
    { label: "Test NDCG@10", value: "0.4948" },
    { label: "Hit Rate@10", value: "0.9058" },
    { label: "Recall@10", value: "0.4793" },
    { label: "Precision@10", value: "0.3130" }
  ]
};

export const profiles = [
  {
    id: "A",
    label: "Usuário A",
    title: "Healthy Shopper",
    icon: "🥬",
    color: "green",
    shortDescription: "Perfil com alta afinidade por orgânicos, frutas, verduras e laticínios naturais.",
    behavior: ["organic", "fresh produce", "reorder friendly", "weekly basket"],
    stats: {
      dominantDepartment: "Produce",
      avgBasket: "8.4 items",
      reorderSignal: "High",
      modelSegment: "Fresh & Organic"
    },
    recommendations: [
      { rank: 1, productId: 13176, name: "Bag of Organic Bananas", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 96, score: 2.84, icon: "🍌", badge: "Top Match" },
      { rank: 2, productId: 21137, name: "Organic Strawberries", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 93, score: 2.71, icon: "🍓", badge: "High Confidence" },
      { rank: 3, productId: 21903, name: "Organic Baby Spinach", category: "Packaged Vegetables", aisleId: 123, departmentId: 4, scorePercent: 90, score: 2.48, icon: "🥬", badge: "Smart Pick" },
      { rank: 4, productId: 47209, name: "Organic Hass Avocado", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 88, score: 2.37, icon: "🥑", badge: "Frequent Pair" },
      { rank: 5, productId: 47766, name: "Organic Avocado", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 86, score: 2.21, icon: "🥑", badge: "Good Match" },
      { rank: 6, productId: 27966, name: "Organic Raspberries", category: "Fresh Fruits", aisleId: 123, departmentId: 4, scorePercent: 84, score: 2.08, icon: "🫐", badge: "Fresh" },
      { rank: 7, productId: 39275, name: "Organic Blueberries", category: "Fresh Fruits", aisleId: 123, departmentId: 4, scorePercent: 81, score: 1.97, icon: "🫐", badge: "Fresh" },
      { rank: 8, productId: 27845, name: "Organic Whole Milk", category: "Milk", aisleId: 84, departmentId: 16, scorePercent: 79, score: 1.82, icon: "🥛", badge: "Basket Add-on" },
      { rank: 9, productId: 33405, name: "Organic Greek Whole Milk Yogurt Vanilla & Lavender", category: "Yogurt", aisleId: 120, departmentId: 16, scorePercent: 76, score: 1.69, icon: "🥣", badge: "Complement" },
      { rank: 10, productId: 26604, name: "Organic Blackberries", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 74, score: 1.57, icon: "🫐", badge: "Nice to Try" }
    ]
  },
  {
    id: "B",
    label: "Usuário B",
    title: "Family Basket",
    icon: "🧺",
    color: "blue",
    shortDescription: "Perfil de cesta familiar, com itens básicos, proteínas, leite, ovos e produtos de rotina.",
    behavior: ["large basket", "household basics", "meal planning", "high reorder"],
    stats: {
      dominantDepartment: "Dairy & Meat",
      avgBasket: "13.2 items",
      reorderSignal: "Very High",
      modelSegment: "Family Essentials"
    },
    recommendations: [
      { rank: 1, productId: 25890, name: "Boneless Skinless Chicken Breasts", category: "Meat Counter", aisleId: 49, departmentId: 12, scorePercent: 95, score: 2.76, icon: "🍗", badge: "Top Match" },
      { rank: 2, productId: 48779, name: "Ground Beef", category: "Meat Counter", aisleId: 122, departmentId: 12, scorePercent: 92, score: 2.58, icon: "🥩", badge: "High Confidence" },
      { rank: 3, productId: 4210, name: "Whole Milk", category: "Milk", aisleId: 84, departmentId: 16, scorePercent: 89, score: 2.39, icon: "🥛", badge: "Routine Item" },
      { rank: 4, productId: 2210, name: "Large Brown Grade AA Eggs", category: "Eggs", aisleId: 86, departmentId: 16, scorePercent: 87, score: 2.26, icon: "🥚", badge: "Staple" },
      { rank: 5, productId: 21497, name: "White Sandwich Bread", category: "Bread", aisleId: 112, departmentId: 3, scorePercent: 85, score: 2.13, icon: "🍞", badge: "Basket Add-on" },
      { rank: 6, productId: 24852, name: "Banana", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 82, score: 1.94, icon: "🍌", badge: "Popular" },
      { rank: 7, productId: 16797, name: "Strawberries", category: "Fresh Produce", aisleId: 24, departmentId: 4, scorePercent: 80, score: 1.86, icon: "🍓", badge: "Fresh" },
      { rank: 8, productId: 6046, name: "Boneless Skinless Chicken Breast", category: "Poultry", aisleId: 35, departmentId: 12, scorePercent: 78, score: 1.74, icon: "🍗", badge: "Meal Prep" },
      { rank: 9, productId: 12881, name: "Natural Cheese Pizza", category: "Cheese", aisleId: 21, departmentId: 16, scorePercent: 75, score: 1.61, icon: "🧀", badge: "Family Night" },
      { rank: 10, productId: 27845, name: "Organic Whole Milk", category: "Milk", aisleId: 84, departmentId: 16, scorePercent: 72, score: 1.49, icon: "🥛", badge: "Alternative" }
    ]
  },
  {
    id: "C",
    label: "Usuário C",
    title: "Snack & Convenience",
    icon: "🍿",
    color: "purple",
    shortDescription: "Perfil com compras rápidas, snacks, bebidas, congelados e produtos de conveniência.",
    behavior: ["snacks", "frozen food", "quick basket", "beverages"],
    stats: {
      dominantDepartment: "Snacks",
      avgBasket: "6.7 items",
      reorderSignal: "Medium",
      modelSegment: "Convenience Picks"
    },
    recommendations: [
      { rank: 1, productId: 1, name: "Chocolate Sandwich Cookies", category: "Cookies & Cakes", aisleId: 61, departmentId: 19, scorePercent: 94, score: 2.69, icon: "🍪", badge: "Top Match" },
      { rank: 2, productId: 141, name: "Restaurant Style Organic Chia & Quinoa Tortilla Chips", category: "Chips & Pretzels", aisleId: 107, departmentId: 19, scorePercent: 91, score: 2.45, icon: "🌮", badge: "High Confidence" },
      { rank: 3, productId: 5258, name: "Sparkling Water", category: "Water & Seltzer", aisleId: 115, departmentId: 7, scorePercent: 89, score: 2.31, icon: "🫧", badge: "Drink Pair" },
      { rank: 4, productId: 18, name: "Pizza for One Suprema Frozen Pizza", category: "Frozen Pizza", aisleId: 79, departmentId: 1, scorePercent: 86, score: 2.12, icon: "🍕", badge: "Quick Meal" },
      { rank: 5, productId: 56, name: "Healthy Pop Butter Popcorn", category: "Popcorn", aisleId: 23, departmentId: 19, scorePercent: 84, score: 2.01, icon: "🍿", badge: "Movie Pick" },
      { rank: 6, productId: 196, name: "Soda", category: "Soft Drinks", aisleId: 77, departmentId: 7, scorePercent: 82, score: 1.91, icon: "🥤", badge: "Beverage" },
      { rank: 7, productId: 764, name: "Classic Butter Microwave Popcorn", category: "Popcorn", aisleId: 23, departmentId: 19, scorePercent: 79, score: 1.78, icon: "🍿", badge: "Snack" },
      { rank: 8, productId: 22545, name: "Pretzels", category: "Chips & Pretzels", aisleId: 107, departmentId: 19, scorePercent: 77, score: 1.67, icon: "🥨", badge: "Crunchy" },
      { rank: 9, productId: 32691, name: "Vanilla Ice Cream", category: "Frozen Dessert", aisleId: 37, departmentId: 1, scorePercent: 74, score: 1.55, icon: "🍦", badge: "Dessert" },
      { rank: 10, productId: 5729, name: "Sea Salt Popcorn", category: "Popcorn", aisleId: 23, departmentId: 19, scorePercent: 71, score: 1.43, icon: "🍿", badge: "Nice to Try" }
    ]
  }
];
