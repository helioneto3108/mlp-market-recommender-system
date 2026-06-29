import { generateBarcodeBars } from '../data'

export default function Barcode({ productId, active }) {
  const bars = generateBarcodeBars(productId)
  const color = active ? '#00ff8888' : '#1a1f2e'

  return (
    <svg
      className="barcode-svg"
      viewBox="0 0 100 30"
      xmlns="http://www.w3.org/2000/svg"
      style={{ width: 100, height: 30, opacity: active ? 1 : 0.5 }}
    >
      {bars.map((b, i) => (
        <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} fill={color} />
      ))}
    </svg>
  )
}
