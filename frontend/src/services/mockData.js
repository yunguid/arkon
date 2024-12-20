export const MOCK_SUMMARY = {
  total_expenses: 12500.75,
  average_daily: 416.69,
  average_monthly: 12500.75,
  daily_expenses: [
    { date: "2024-03-01", amount: 425.50 },
    { date: "2024-03-02", amount: 380.25 },
    // Add more mock daily data
  ],
  top_5_expenses: [
    { description: "RENT PAYMENT", amount: 2000.00 },
    { description: "CAR PAYMENT", amount: 450.00 },
    // Add more mock expenses
  ],
  ai_categories: [
    { main_category: "Housing", sub_category: "Rent", detail_category: "Monthly", amount: 2000.00, count: 1 },
    { main_category: "Transport", sub_category: "Car", detail_category: "Payment", amount: 450.00, count: 1 },
    // Add more categories
  ]
};

export const mockApi = {
  uploadFile: () => Promise.resolve({ summary: MOCK_SUMMARY, file_id: "mock-1" }),
  getFiles: () => Promise.resolve([
    { id: "mock-1", filename: "March_2024.csv", upload_date: "2024-03-19T00:00:00Z" }
  ]),
  getFile: () => Promise.resolve({ summary: MOCK_SUMMARY }),
  getStockPrice: () => Promise.resolve({ price: 180.25 }),
  getWatchlist: () => Promise.resolve({ 
    watchlist: [
      { symbol: "AAPL", added_date: "2024-03-19T00:00:00Z" },
      { symbol: "MSFT", added_date: "2024-03-19T00:00:00Z" }
    ]
  }),
  addToWatchlist: (symbol) => Promise.resolve({ status: 'success' }),
  analyzeWatchlist: () => Promise.resolve({ status: 'complete' }),
  uploadStockData: () => Promise.resolve({
    summary: {
      daily_prices: [
        { date: '2024-03-01', price: 180.25 },
        { date: '2024-03-02', price: 182.50 },
        { date: '2024-03-03', price: 185.75 }
      ]
    }
  }),
  getNews: (symbol) => Promise.resolve({
    news: [
      {
        date: "2024-03-19T00:00:00Z",
        title: "Mock News Article",
        summary: "This is a mock news summary for testing purposes",
        sentiment: 0.75
      }
    ]
  }),
}; 