import { useEffect, useState } from "react";
import {
  API_USER_URL,
  buildProfileFromApiUser,
  fallbackProfile,
  modelStats
} from "./data/mockRecommendations";

function Header() {
  return (
    <header className="hero">
      <div className="hero__copy">
        <div className="eyebrow">
          <span className="pulse-dot" /> API Connected View
        </div>
        <h1>Market Recommender</h1>
        <p>
          Front-end dark para visualizar as recomendações do modelo para um único usuário.
          A tela busca os dados em uma API local e transforma os produtos em cards visuais.
        </p>
      </div>

      <div className="hero__panel">
        <div className="mini-orbit">
          <span>🛒</span>
          <i />
          <i />
          <i />
        </div>
        <div>
          <strong>{modelStats.name}</strong>
          <small>{modelStats.description}</small>
        </div>
      </div>
    </header>
  );
}

function UserPanel({ profile, apiStatus }) {
  return (
    <section className="panel selector-panel">
      <div className="section-title">
        <span>01</span>
        <h2>Selected customer</h2>
      </div>

      <div className={`profile-card profile-card--${profile.color} is-active single-user-card`}>
        <div className="profile-card__icon">{profile.icon}</div>
        <div className="profile-card__text">
          <strong>{profile.label}</strong>
          <span>{profile.title}</span>
        </div>
        <div className="profile-card__arrow">✓</div>
      </div>

      <div className={`api-status api-status--${apiStatus.type}`}>
        <strong>{apiStatus.title}</strong>
        <span>{apiStatus.message}</span>
      </div>

      <div className="api-url-card">
        <small>API endpoint</small>
        <code>{API_USER_URL}</code>
      </div>
    </section>
  );
}

function ModelPanel() {
  return (
    <section className="panel model-panel">
      <div className="section-title">
        <span>02</span>
        <h2>Model snapshot</h2>
      </div>

      <div className="model-chip-grid">
        <div className="model-chip">
          <small>Primary metric</small>
          <strong>{modelStats.primaryMetric}</strong>
        </div>
        <div className="model-chip">
          <small>Top K</small>
          <strong>{modelStats.primaryK}</strong>
        </div>
        <div className="model-chip">
          <small>Embedding dim</small>
          <strong>{modelStats.embeddingDim}</strong>
        </div>
        <div className="model-chip">
          <small>Hidden layers</small>
          <strong>{modelStats.hiddenLayers}</strong>
        </div>
      </div>

      <div className="metric-grid">
        {modelStats.metrics.map((metric) => (
          <div className="metric-card" key={metric.label}>
            <small>{metric.label}</small>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function LoadingSimulation() {
  return (
    <div className="loading-card">
      <div className="scanner">
        <span />
      </div>
      <strong>Loading user recommendations...</strong>
      <p>Requesting the API, reading products and building the visual recommendation ranking.</p>
    </div>
  );
}

function RecommendationCard({ item }) {
  const metadata = [
    item.category,
    item.aisleId ? `aisle ${item.aisleId}` : null,
    item.departmentId ? `department ${item.departmentId}` : null
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <article className="recommendation-card">
      <div className="recommendation-card__rank">#{item.rank}</div>

      <div className="product-illustration" aria-hidden="true">
        <div className="product-illustration__glow" />
        <span>{item.icon}</span>
      </div>

      <div className="recommendation-card__body">
        <div className="recommendation-card__topline">
          <span>{item.badge}</span>
          <small>ID {item.productId}</small>
        </div>

        <h3>{item.name}</h3>
        <p>{metadata}</p>

        <div className="product-signal-grid">
          <div>
            <small>Purchase count</small>
            <strong>{item.purchaseCount}</strong>
          </div>
          <div>
            <small>Reorder rate</small>
            <strong>{Number(item.reorderRate).toFixed(2)}</strong>
          </div>
        </div>

        <div className="score-row">
          <div className="score-bar" aria-label={`Visual recommendation score ${item.scorePercent}%`}>
            <span style={{ width: `${item.scorePercent}%` }} />
          </div>
          <strong>{item.scorePercent}%</strong>
        </div>

        <button className="basket-button" type="button">
          View product
        </button>
      </div>
    </article>
  );
}

function Recommendations({ profile, isRunning, onRun }) {
  const bestPick = profile.recommendations[0];

  return (
    <section className="panel recommendations-panel">
      <div className="recommendations-header">
        <div>
          <div className="section-title section-title--inline">
            <span>03</span>
            <h2>User products</h2>
          </div>
          <p>
            Products returned for <strong>{profile.label}</strong> by the local API.
          </p>
        </div>

        <button className="run-button" onClick={onRun} disabled={isRunning} type="button">
          {isRunning ? "Loading API..." : "Refresh from API"}
        </button>
      </div>

      {isRunning ? (
        <LoadingSimulation />
      ) : (
        <>
          {bestPick ? (
            <div className="best-pick-card">
              <div className="best-pick-card__icon">{bestPick.icon}</div>
              <div>
                <span>Best visual match</span>
                <strong>{bestPick.name}</strong>
                <small>{bestPick.scorePercent}% display score</small>
              </div>
            </div>
          ) : (
            <div className="empty-card">No products returned by the API.</div>
          )}

          <div className="recommendation-grid">
            {profile.recommendations.map((item) => (
              <RecommendationCard key={item.productId} item={item} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function RankingTable({ profile }) {
  return (
    <section className="panel ranking-panel">
      <div className="section-title">
        <span>04</span>
        <h2>Product table</h2>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Product</th>
              <th>Category</th>
              <th>Purchase count</th>
              <th>Display score</th>
            </tr>
          </thead>
          <tbody>
            {profile.recommendations.map((item) => (
              <tr key={item.productId}>
                <td>#{item.rank}</td>
                <td>
                  <span className="table-product-icon">{item.icon}</span>
                  {item.name}
                </td>
                <td>{item.category}</td>
                <td>{item.purchaseCount}</td>
                <td>{item.scorePercent}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function App() {
  const [profile, setProfile] = useState(fallbackProfile);
  const [isRunning, setIsRunning] = useState(false);
  const [apiStatus, setApiStatus] = useState({
    type: "idle",
    title: "Ready",
    message: "Click Refresh from API or start the backend to load live data."
  });

  async function loadUserFromApi() {
    setIsRunning(true);
    setApiStatus({
      type: "loading",
      title: "Loading",
      message: "Trying to connect with the local API..."
    });

    try {
      const response = await fetch(API_USER_URL);

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data = await response.json();
      const nextProfile = buildProfileFromApiUser(data);

      setProfile(nextProfile);
      setApiStatus({
        type: "success",
        title: "API connected",
        message: `Loaded user ${data.user_id} with ${nextProfile.recommendations.length} products.`
      });
    } catch (error) {
      console.error(error);
      setProfile(fallbackProfile);
      setApiStatus({
        type: "error",
        title: "Using local fallback",
        message: "Could not reach the API. Check if the backend is running on port 8010 and if CORS is enabled."
      });
    } finally {
      setIsRunning(false);
    }
  }

  useEffect(() => {
    loadUserFromApi();
  }, []);

  return (
    <main className="app-shell">
      <div className="background-orb background-orb--one" />
      <div className="background-orb background-orb--two" />

      <Header />

      <div className="dashboard-layout">
        <aside className="sidebar">
          <UserPanel profile={profile} apiStatus={apiStatus} />
          <ModelPanel />
        </aside>

        <div className="content-stack">
          <Recommendations profile={profile} isRunning={isRunning} onRun={loadUserFromApi} />
          <RankingTable profile={profile} />
        </div>
      </div>
    </main>
  );
}

export default App;
