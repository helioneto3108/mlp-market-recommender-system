import { formatPrice } from '../data'

export default function BeltPanel({ cart, products, onRemove }) {
  const total = cart.reduce((sum, id) => {
    const p = products.find((x) => x.id === id)
    return sum + (p?.price ?? 0)
  }, 0)

  return (
    <div className="middle-col">
      <div className="belt-area">
        <div className="belt-label">Itens escaneados</div>
        <div className="belt-surface">
          <div className="belt-stripes" />
          {cart.length === 0 ? (
            <div className="belt-empty">
              <i className="ti ti-barcode" aria-hidden="true" />
              Nenhum item escaneado
            </div>
          ) : (
            cart.map((id) => {
              const p = products.find((x) => x.id === id)
              if (!p) return null
              return (
                <div key={id} className="cart-row">
                  <span className="cr-emoji">{p.emoji}</span>
                  <span className="cr-name">{p.name}</span>
                  <span className="cr-price">{formatPrice(p.price)}</span>
                  <button
                    className="cr-rm"
                    onClick={() => onRemove(id)}
                    aria-label={`Remover ${p.name}`}
                  >
                    ×
                  </button>
                </div>
              )
            })
          )}
        </div>
      </div>

      <div className="total-bar">
        <div className="total-label">Total</div>
        <div className="total-val">{formatPrice(total)}</div>
      </div>
    </div>
  )
}
