import { useMemo, useState } from "react";
import { modelStats, profiles } from "./data/mockRecommendations";

function Header() {
  return (
    <header className="hero">
      <div className="hero__copy">
        <div className="eyebrow">
          <span className="pulse-dot" /> ML Model Online
        </div>
        <h1>Market Recommender</h1>
        <p>
          Simulação visual de um sistema de recomendação para mercado, usando perfis de compra,
          embeddings e ranking top-k para sugerir produtos ao usuário.
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

function ProfileSelector({ selectedProfile, onSelect }) {
  return (
    <section className="panel selector-panel">
      <div className="section-title">
        <span>01</span>
        <h2>Choose customer profile</h2>
      </div>

      <div className="profile-list">
        {profiles.map((profile) => {
          const isActive = profile.id === selectedProfile.id;

          return (
            <button
              key={profile.id}
              className={`profile-card profile-card--${profile.color} ${isActive ? "is-active" : ""}`}
              onClick={() => onSelect(profile.id)}
              type="button"
            >
              <div className="profile-card__icon">{profile.icon}</div>
              <div className="profile-card__text">
                <strong>{profile.label}</strong>
                <span>{profile.title}</span>
              </div>
              <div className="profile-card__arrow">›</div>
            </button>
          );
        })}
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

function ProfileSummary({ profile }) {
  return (
    <section className={`panel summary-card summary-card--${profile.color}`}>
      <div>
        <div className="section-title section-title--inline">
          <span>03</span>
          <h2>{profile.label}</h2>
        </div>
        <h3>{profile.title}</h3>
        <p>{profile.shortDescription}</p>
      </div>

      <div className="behavior-list">
        {profile.behavior.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>

      <div className="summary-grid">
        <div>
          <small>Dominant department</small>
          <strong>{profile.stats.dominantDepartment}</strong>
        </div>
        <div>
          <small>Average basket</small>
          <strong>{profile.stats.avgBasket}</strong>
        </div>
        <div>
          <small>Reorder signal</small>
          <strong>{profile.stats.reorderSignal}</strong>
        </div>
        <div>
          <small>Model segment</small>
          <strong>{profile.stats.modelSegment}</strong>
        </div>
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
      <strong>Ranking candidate products...</strong>
      <p>Applying user embeddings, product affinity and temporal behavior signals.</p>
    </div>
  );
}

function RecommendationCard({ item }) {
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
        <p>
          {item.category} · aisle {item.aisleId} · department {item.departmentId}
        </p>

        <div className="score-row">
          <div className="score-bar" aria-label={`Recommendation score ${item.scorePercent}%`}>
            <span style={{ width: `${item.scorePercent}%` }} />
          </div>
          <strong>{item.scorePercent}%</strong>
        </div>

        <button className="basket-button" type="button">
          Add to basket
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
            <span>04</span>
            <h2>Top 10 recommendations</h2>
          </div>
          <p>
            Ranking for <strong>{profile.label}</strong> based on simulated model outputs.
          </p>
        </div>

        <button className="run-button" onClick={onRun} disabled={isRunning} type="button">
          {isRunning ? "Running model..." : "Run recommendation"}
        </button>
      </div>

      {isRunning ? (
        <LoadingSimulation />
      ) : (
        <>
          <div className="best-pick-card">
            <div className="best-pick-card__icon">{bestPick.icon}</div>
            <div>
              <span>Best match</span>
              <strong>{bestPick.name}</strong>
              <small>{bestPick.scorePercent}% recommendation score</small>
            </div>
          </div>

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
        <span>05</span>
        <h2>Ranking table</h2>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Product</th>
              <th>Category</th>
              <th>Score</th>
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
  const [selectedProfileId, setSelectedProfileId] = useState("A");
  const [isRunning, setIsRunning] = useState(false);

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.id === selectedProfileId) ?? profiles[0],
    [selectedProfileId]
  );

  function handleSelectProfile(profileId) {
    setSelectedProfileId(profileId);
    setIsRunning(true);
    window.setTimeout(() => setIsRunning(false), 700);
  }

  function handleRunRecommendation() {
    setIsRunning(true);
    window.setTimeout(() => setIsRunning(false), 900);
  }

  return (
    <main className="app-shell">
      <div className="background-orb background-orb--one" />
      <div className="background-orb background-orb--two" />

      <Header />

      <div className="dashboard-layout">
        <aside className="sidebar">
          <ProfileSelector selectedProfile={selectedProfile} onSelect={handleSelectProfile} />
          <ModelPanel />
        </aside>

        <div className="content-stack">
          <ProfileSummary profile={selectedProfile} />
          <Recommendations
            profile={selectedProfile}
            isRunning={isRunning}
            onRun={handleRunRecommendation}
          />
          <RankingTable profile={selectedProfile} />
        </div>
      </div>
    </main>
  );
}

export default App;
