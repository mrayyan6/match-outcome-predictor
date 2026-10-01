// Kit colours for the dot next to a team: shirt, then the second colour.
// Keyed by football-data.co.uk's names, which is what the data uses
// internally. Picked by hand, close enough rather than exact.

const KIT = {
  // Premier League
  "Arsenal": ["#ef0107", "#ffffff"],
  "Aston Villa": ["#670e36", "#95bfe5"],
  "Bournemouth": ["#da291c", "#111111"],
  "Brentford": ["#e30613", "#ffffff"],
  "Brighton": ["#0057b8", "#ffffff"],
  "Burnley": ["#6c1d45", "#99d6ea"],
  "Chelsea": ["#034694", "#ffffff"],
  "Coventry": ["#59cbe8", "#ffffff"],
  "Crystal Palace": ["#1b458f", "#c4122e"],
  "Everton": ["#003399", "#ffffff"],
  "Fulham": ["#ffffff", "#111111"],
  "Hull": ["#f5a12d", "#111111"],
  "Ipswich": ["#3a64a3", "#ffffff"],
  "Leeds": ["#ffffff", "#1d428a"],
  "Liverpool": ["#c8102e", "#c8102e"],
  "Man City": ["#6cabdd", "#ffffff"],
  "Man United": ["#da291c", "#111111"],
  "Newcastle": ["#111111", "#ffffff"],
  "Nott'm Forest": ["#dd0000", "#ffffff"],
  "Sunderland": ["#eb172b", "#ffffff"],
  "Tottenham": ["#ffffff", "#132257"],
  "West Ham": ["#7a263a", "#1bb1e7"],
  "Wolves": ["#fdb913", "#231f20"],
  // La Liga
  "Alaves": ["#0761af", "#ffffff"],
  "Ath Bilbao": ["#ee2523", "#ffffff"],
  "Ath Madrid": ["#cb3524", "#272e61"],
  "Barcelona": ["#004d98", "#a50044"],
  "Betis": ["#00954c", "#ffffff"],
  "Celta": ["#8ac3ee", "#ffffff"],
  "Elche": ["#ffffff", "#05642c"],
  "Espanol": ["#007fc8", "#ffffff"],
  "Getafe": ["#005999", "#005999"],
  "Girona": ["#cd2534", "#ffffff"],
  "La Coruna": ["#1b4e9b", "#ffffff"],
  "Levante": ["#b4053f", "#004b97"],
  "Malaga": ["#3a87c8", "#ffffff"],
  "Mallorca": ["#e20613", "#111111"],
  "Osasuna": ["#d91a21", "#0a346f"],
  "Oviedo": ["#0055a4", "#ffffff"],
  "Real Madrid": ["#ffffff", "#febe10"],
  "Santander": ["#ffffff", "#00a650"],
  "Sevilla": ["#ffffff", "#d71920"],
  "Sociedad": ["#0067b1", "#ffffff"],
  "Valencia": ["#ffffff", "#111111"],
  "Vallecano": ["#ffffff", "#e53027"],
  "Villarreal": ["#ffe114", "#005187"],
};

export function clubDot(team) {
  const dot = document.createElement("i");
  dot.className = "kit";
  dot.setAttribute("aria-hidden", "true");
  const colours = KIT[team];
  if (colours) {
    dot.style.setProperty("--kit-a", colours[0]);
    dot.style.setProperty("--kit-b", colours[1]);
  }
  return dot;
}
