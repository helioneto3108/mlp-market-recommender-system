import { useEffect, useRef } from 'react'
import Barcode from './Barcode'
import { formatPrice } from '../data'

export default function ScannerPanel({ products, cart, scanning, lastScanned, onScan }) {
  const scanLineRef = useRef(null)

  return (
    <div className="scanner-col">
      {/* Screen */}
      <div className="scanner-screen">
        <div className="scanner-border" />
        <div className="scanner-border2" />
        <div className={`scan-line ${scanning ? 'active' : ''}`} />

        {scanning && <div className="beep-ring go" key={Date.now()} />}

        <div className="scan-product-display">
          {lastScanned ? (
            <>
              <div className="scan-emoji-wrap">
                <span style={{ fontSize: 38 }}>{lastScanned.emoji}</span>
                <div className="scan-glow lit" />
              </div>
              <div className="scan-name">{lastScanned.name}</div>
              <div className="scan-price-disp">{formatPrice(lastScanned.price)}</div>
              <Barcode productId={lastScanned.id} active />
            </>
          ) : (
            <>
              <div className="scan-label">Aguardando produto...</div>
              <Barcode productId={0} active={false} />
            </>
          )}
        </div>
      </div>

      {/* Shelf */}
      <div className="shelf">
        <div className="shelf-label">Produtos — clique para escanear</div>
        <div className="shelf-grid">
          {products.map((p) => (
            <button
              key={p.id}
              className={`shelf-item ${cart.includes(p.id) ? 'in-cart' : ''}`}
              onClick={() => onScan(p.id)}
              disabled={cart.includes(p.id) || scanning}
              aria-label={`Escanear ${p.name}`}
            >
              <span className="se">{p.emoji}</span>
              <span className="sn">{p.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
