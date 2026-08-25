// Fetch market prices
useEffect(() => {
  axios.get('http://your-backend/market_prices?crop=rice').then(res => {
    setPrices(res.data);  // Current price, trend, yield estimate
  });
}, []);
