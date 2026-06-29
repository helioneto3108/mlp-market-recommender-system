import { useEffect, useRef } from 'react'
import { formatPrice } from '../data'

export default function ReceiptPanel({ cart, products, recommendations }) {
  const wrapRef = useRef(null)
  const total = cart.reduce((sum, id) => {
    const p = products.find((x) => x.id === id)
    return sum + (p?.price ?? 0)
  }, 0)

  useEffect(() => {
    if (wrapRef.current) {
      wrapRef.current.scrollTop = wrapRef.current.scrollHeight
    }
  }, [cart, recommendations])

  const dateStr = new Date().toLocaleString('pt-BR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className="right-col">
      <div className="receipt-wrap" ref={wrapRef}>
        <div className="r-store">MERCADO ML</div>
        <div className="r-sub">{dateStr}</div>
        <div className="r-div" />

        {cart.length === 0 ? (
          <div className="r-empty">Sem itens</div>
        ) : (
          <>
            <div className="r-section">Itens</div>
            {cart.map((id) => {
              const p = products.find((x) => x.id === id)
              if (!p) return null
              return (
                <div key={id} className="r-row bought">
                  <span>{p.emoji} {p.name}</span>
                  <span>{formatPrice(p.price)}</span>
                </div>
              )
            })}

            {recommendations.length > 0 && (
              <>
                <div className="r-div" />
                <div className="r-section">⚡ Recomendações</div>
                {recommendations.map((rec) => {
                  const p = products.find((x) => x.id === rec.id)
                  if (!p) return null
                  return (
                    <div key={rec.id} className="rec-block">
                      <div className="rec-line">
                        <span className="rec-nm">{p.emoji} {p.name}</span>
                        <span className="rec-pc">{rec.p}%</span>
                      </div>
                      <div className="rbar">
                        <div className="rbar-f" style={{ width: `${rec.p}%` }} />
                      </div>
                    </div>
                  )
                })}
              </>
            )}
          </>
        )}

        <div className="r-div" />
        <div className="r-total-row">
          <span>TOTAL</span>
          <span>{formatPrice(total)}</span>
        </div>
      </div>
    </div>
  )
}
