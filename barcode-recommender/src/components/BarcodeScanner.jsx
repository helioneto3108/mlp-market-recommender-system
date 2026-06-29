import { useState, useEffect, useCallback } from 'react'
import ScannerPanel from './ScannerPanel'
import BeltPanel from './BeltPanel'
import ReceiptPanel from './ReceiptPanel'
import { PRODUCTS, fetchRecommendations } from '../data'
import './BarcodeScanner.css'

export default function BarcodeScanner() {
  const [cart, setCart] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [scanning, setScanning] = useState(false)
  const [lastScanned, setLastScanned] = useState(null)
  const [time, setTime] = useState('')

  // Clock
  useEffect(() => {
    const update = () =>
      setTime(
        new Date().toLocaleTimeString('pt-BR', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      )

    update()

    const id = setInterval(update, 1000)

    return () => clearInterval(id)
  }, [])

  // Busca recomendações reais na API sempre que o carrinho mudar
  useEffect(() => {
    if (cart.length === 0) {
      setRecommendations([])
      return
    }

    fetchRecommendations(cart)
      .then((data) => {
        setRecommendations(data)
      })
      .catch((error) => {
        console.error('Erro ao buscar recomendações:', error)
        setRecommendations([])
      })
  }, [cart])

  const handleScan = useCallback(
    (productId) => {
      if (scanning || cart.includes(productId)) return

      setScanning(true)

      const product = PRODUCTS.find((p) => p.id === productId)

      setTimeout(() => {
        setLastScanned(product)
        setCart((prev) => [...prev, productId])
        setScanning(false)

        setTimeout(() => setLastScanned(null), 2000)
      }, 700)
    },
    [scanning, cart]
  )

  const handleRemove = useCallback((productId) => {
    setCart((prev) => prev.filter((id) => id !== productId))
  }, [])

  return (
    <div className="scanner-app">
      {/* Top bar */}
      <div className="topbar">
        <div className="tb-lights">
          <div className="tbl" style={{ background: '#ff5f57' }} />
          <div className="tbl" style={{ background: '#febc2e' }} />
          <div className="tbl" style={{ background: '#28c840' }} />
        </div>

        <div className="tb-title">Caixa 04 — Mercado ML</div>

        <div className="tb-time">{time}</div>
      </div>

      {/* Main layout */}
      <div className="layout">
        <ScannerPanel
          products={PRODUCTS}
          cart={cart}
          scanning={scanning}
          lastScanned={lastScanned}
          onScan={handleScan}
        />

        <BeltPanel
          cart={cart}
          products={PRODUCTS}
          onRemove={handleRemove}
        />

        <ReceiptPanel
          cart={cart}
          products={PRODUCTS}
          recommendations={recommendations}
        />
      </div>
    </div>
  )
}