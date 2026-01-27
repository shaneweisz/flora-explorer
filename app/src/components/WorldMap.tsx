"use client";

import React, { useState, useEffect, memo } from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  ZoomableGroup,
} from "react-simple-maps";

// Using the recommended TopoJSON from react-simple-maps
const GEO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

// Country name (from TopoJSON) to ISO 3166-1 alpha-2 mapping for GBIF
const NAME_TO_ALPHA2: Record<string, string> = {
  "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Angola": "AO", "Argentina": "AR",
  "Armenia": "AM", "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ", "Bangladesh": "BD",
  "Belarus": "BY", "Belgium": "BE", "Benin": "BJ", "Bhutan": "BT", "Bolivia": "BO",
  "Bosnia and Herz.": "BA", "Botswana": "BW", "Brazil": "BR", "Brunei": "BN", "Bulgaria": "BG",
  "Burkina Faso": "BF", "Burundi": "BI", "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA",
  "Central African Rep.": "CF", "Chad": "TD", "Chile": "CL", "China": "CN", "Colombia": "CO",
  "Congo": "CG", "Dem. Rep. Congo": "CD", "Costa Rica": "CR", "Côte d'Ivoire": "CI",
  "Croatia": "HR", "Cuba": "CU", "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK",
  "Djibouti": "DJ", "Dominican Rep.": "DO", "Ecuador": "EC", "Egypt": "EG", "El Salvador": "SV",
  "Eq. Guinea": "GQ", "Eritrea": "ER", "Estonia": "EE", "eSwatini": "SZ", "Ethiopia": "ET",
  "Fiji": "FJ", "Finland": "FI", "France": "FR", "Gabon": "GA", "Gambia": "GM", "Georgia": "GE",
  "Germany": "DE", "Ghana": "GH", "Greece": "GR", "Greenland": "GL", "Guatemala": "GT",
  "Guinea": "GN", "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT", "Honduras": "HN",
  "Hungary": "HU", "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
  "Ireland": "IE", "Israel": "IL", "Italy": "IT", "Jamaica": "JM", "Japan": "JP", "Jordan": "JO",
  "Kazakhstan": "KZ", "Kenya": "KE", "North Korea": "KP", "South Korea": "KR", "Kuwait": "KW",
  "Kyrgyzstan": "KG", "Laos": "LA", "Latvia": "LV", "Lebanon": "LB", "Lesotho": "LS",
  "Liberia": "LR", "Libya": "LY", "Lithuania": "LT", "Luxembourg": "LU", "Madagascar": "MG",
  "Malawi": "MW", "Malaysia": "MY", "Mali": "ML", "Mauritania": "MR", "Mexico": "MX",
  "Moldova": "MD", "Mongolia": "MN", "Montenegro": "ME", "Morocco": "MA", "Mozambique": "MZ",
  "Myanmar": "MM", "Namibia": "NA", "Nepal": "NP", "Netherlands": "NL", "New Zealand": "NZ",
  "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG", "Norway": "NO", "Oman": "OM",
  "Pakistan": "PK", "Panama": "PA", "Papua New Guinea": "PG", "Paraguay": "PY", "Peru": "PE",
  "Philippines": "PH", "Poland": "PL", "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA",
  "Romania": "RO", "Russia": "RU", "Rwanda": "RW", "Saudi Arabia": "SA", "Senegal": "SN",
  "Serbia": "RS", "Sierra Leone": "SL", "Singapore": "SG", "Slovakia": "SK", "Slovenia": "SI",
  "Solomon Is.": "SB", "Somalia": "SO", "South Africa": "ZA", "S. Sudan": "SS", "Spain": "ES",
  "Sri Lanka": "LK", "Sudan": "SD", "Suriname": "SR", "Sweden": "SE", "Switzerland": "CH",
  "Syria": "SY", "Taiwan": "TW", "Tajikistan": "TJ", "Tanzania": "TZ", "Thailand": "TH",
  "Timor-Leste": "TL", "Togo": "TG", "Trinidad and Tobago": "TT", "Tunisia": "TN",
  "Turkey": "TR", "Turkmenistan": "TM", "Uganda": "UG", "Ukraine": "UA",
  "United Arab Emirates": "AE", "United Kingdom": "GB", "United States of America": "US",
  "Uruguay": "UY", "Uzbekistan": "UZ", "Vanuatu": "VU", "Venezuela": "VE", "Vietnam": "VN",
  "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW", "Palestine": "PS", "Kosovo": "XK",
  "North Macedonia": "MK", "New Caledonia": "NC", "W. Sahara": "EH", "Fr. S. Antarctic Lands": "TF",
  "Falkland Is.": "FK",
};

// Complete ISO 3166-1 alpha-2 to country name mapping (for display)
// Includes all countries, territories, and small island nations
export const ALPHA2_TO_NAME: Record<string, string> = {
  // From TopoJSON (use these names for map consistency)
  ...Object.fromEntries(Object.entries(NAME_TO_ALPHA2).map(([name, code]) => [code, name])),
  // Additional countries and territories not in TopoJSON
  "AD": "Andorra", "AG": "Antigua and Barbuda", "AI": "Anguilla", "AQ": "Antarctica",
  "AS": "American Samoa", "AW": "Aruba", "AX": "Åland Islands", "BB": "Barbados",
  "BH": "Bahrain", "BL": "Saint Barthélemy", "BM": "Bermuda", "BQ": "Bonaire",
  "BS": "Bahamas", "BV": "Bouvet Island", "BZ": "Belize", "CC": "Cocos Islands",
  "CK": "Cook Islands", "CV": "Cape Verde", "CW": "Curaçao", "CX": "Christmas Island",
  "DM": "Dominica", "FK": "Falkland Islands", "FM": "Micronesia", "FO": "Faroe Islands",
  "GD": "Grenada", "GF": "French Guiana", "GG": "Guernsey", "GI": "Gibraltar",
  "GP": "Guadeloupe", "GS": "South Georgia", "GU": "Guam", "HK": "Hong Kong",
  "HM": "Heard Island", "IM": "Isle of Man", "IO": "British Indian Ocean Territory",
  "JE": "Jersey", "KI": "Kiribati", "KM": "Comoros", "KN": "Saint Kitts and Nevis",
  "KY": "Cayman Islands", "LC": "Saint Lucia", "LI": "Liechtenstein", "MC": "Monaco",
  "MF": "Saint Martin", "MH": "Marshall Islands", "MO": "Macao", "MP": "Northern Mariana Islands",
  "MQ": "Martinique", "MS": "Montserrat", "MT": "Malta", "MU": "Mauritius", "MV": "Maldives",
  "NF": "Norfolk Island", "NR": "Nauru", "NU": "Niue", "PF": "French Polynesia",
  "PM": "Saint Pierre and Miquelon", "PN": "Pitcairn", "PW": "Palau", "RE": "Réunion",
  "SC": "Seychelles", "SH": "Saint Helena", "SJ": "Svalbard", "SM": "San Marino",
  "ST": "São Tomé and Príncipe", "SV": "El Salvador", "SX": "Sint Maarten",
  "TC": "Turks and Caicos", "TK": "Tokelau", "TO": "Tonga", "TV": "Tuvalu",
  "UM": "U.S. Minor Outlying Islands", "VA": "Vatican City", "VC": "Saint Vincent and the Grenadines",
  "VG": "British Virgin Islands", "VI": "U.S. Virgin Islands", "WF": "Wallis and Futuna",
  "WS": "Samoa", "YT": "Mayotte",
};

interface CountryStats {
  [countryCode: string]: {
    occurrences: number;
    species: number;
  };
}

// Color scale for heatmap: pale green -> medium green -> dark green
function getHeatmapColor(value: number, maxValue: number): string {
  if (value === 0 || maxValue === 0) return "#f5f5f4"; // stone-100

  // Use log scale with high power to push most countries to pale end
  // Higher power = more countries appear pale, only highest values get dark
  const logValue = Math.log10(value + 1);
  const logMax = Math.log10(maxValue + 1);
  const ratio = Math.pow(logValue / logMax, 2.0);

  // Color scale: #dcfce7 (green-100) -> #86efac (green-300) -> #22c55e (green-500) -> #166534 (green-800)
  if (ratio < 0.33) {
    // Pale green to light green
    const t = ratio * 3;
    const r = Math.round(220 + (134 - 220) * t);
    const g = Math.round(252 + (239 - 252) * t);
    const b = Math.round(231 + (172 - 231) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (ratio < 0.66) {
    // Light green to medium green
    const t = (ratio - 0.33) * 3;
    const r = Math.round(134 + (34 - 134) * t);
    const g = Math.round(239 + (197 - 239) * t);
    const b = Math.round(172 + (94 - 172) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Medium green to dark green
    const t = (ratio - 0.66) * 3;
    const r = Math.round(34 + (22 - 34) * t);
    const g = Math.round(197 + (101 - 197) * t);
    const b = Math.round(94 + (52 - 94) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
  if (num >= 1000) return (num / 1000).toFixed(1) + "K";
  return num.toString();
}

interface WorldMapProps {
  selectedCountry: string | null;
  onCountrySelect: (countryCode: string, countryName: string) => void;
  onClearSelection: () => void;
  selectedTaxon?: string | null;
  // Optional pre-computed stats (for Red List - avoids API call)
  precomputedStats?: CountryStats;
  // Label for the stat shown in tooltip (default: "Occurrences")
  statLabel?: string;
}

function WorldMap({ selectedCountry, onCountrySelect, onClearSelection, selectedTaxon, precomputedStats, statLabel = "Occurrences" }: WorldMapProps) {
  const [hoveredCountry, setHoveredCountry] = useState<string | null>(null);
  const [hoveredCountryCode, setHoveredCountryCode] = useState<string | null>(null);
  const [countryStats, setCountryStats] = useState<CountryStats>({});
  const [loading, setLoading] = useState(true);

  // Use precomputed stats if provided, otherwise fetch from API
  useEffect(() => {
    if (precomputedStats) {
      setCountryStats(precomputedStats);
      setLoading(false);
      return;
    }

    if (!selectedTaxon) {
      setCountryStats({});
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`/api/country/stats?taxon=${encodeURIComponent(selectedTaxon)}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.stats) {
          setCountryStats(data.stats);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedTaxon, precomputedStats]);

  // Calculate max value for heatmap scaling
  const maxOccurrences = Object.values(countryStats).reduce(
    (max, stat) => Math.max(max, stat.occurrences),
    0
  );

  const getCountryColor = (alpha2: string | undefined, isSelected: boolean): string => {
    if (isSelected) return "#22c55e";
    if (!alpha2) return "#f4f4f5";

    const stats = countryStats[alpha2];
    if (!stats) return "#f4f4f5";

    return getHeatmapColor(stats.occurrences, maxOccurrences);
  };

  const hoveredStats = hoveredCountryCode ? countryStats[hoveredCountryCode] : null;

  return (
    <div className="relative bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-3 h-full flex flex-col">
      {/* Header with controls and legend */}
      <div className="flex items-center justify-between mb-0">
        <div>
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Explore by Country
          </h2>
          <p className="text-xs text-zinc-500">
            {selectedCountry
              ? `${ALPHA2_TO_NAME[selectedCountry] || selectedCountry} · Click to deselect`
              : "Click a country to filter"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Legend in header */}
          <div className="flex items-center gap-1 text-[10px]">
            <span className="text-zinc-400">Low</span>
            <div
              className="w-12 h-1.5 rounded"
              style={{
                background: "linear-gradient(to right, #dcfce7, #86efac, #22c55e, #166534)"
              }}
            />
            <span className="text-zinc-400">High</span>
          </div>
          {selectedCountry && (
            <button
              onClick={onClearSelection}
              className="px-2 py-1 text-xs text-zinc-600 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200 bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 rounded transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredCountry && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 z-10 bg-white dark:bg-zinc-800 px-3 py-2 rounded-lg shadow-lg text-sm text-zinc-700 dark:text-zinc-300 pointer-events-none border border-zinc-200 dark:border-zinc-700 min-w-[140px]">
          <div className="font-medium text-zinc-900 dark:text-zinc-100">{hoveredCountry}</div>
          {hoveredStats ? (
            <div className="mt-1 space-y-0.5">
              <div className="flex justify-between gap-4 text-xs">
                <span className="text-zinc-500">{statLabel}</span>
                <span className="font-medium text-zinc-700 dark:text-zinc-300 tabular-nums">{formatNumber(hoveredStats.occurrences)}</span>
              </div>
              {hoveredStats.species > 0 && (
                <div className="flex justify-between gap-4 text-xs">
                  <span className="text-zinc-500">Species</span>
                  <span className="font-medium text-zinc-700 dark:text-zinc-300 tabular-nums">{formatNumber(hoveredStats.species)}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-zinc-400 mt-1">No data available</div>
          )}
        </div>
      )}

      {/* Map */}
      <div className="flex-1 rounded-lg overflow-hidden relative" style={{ minHeight: "200px" }}>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-zinc-900/50 z-10">
            <div className="text-sm text-zinc-500">Loading heatmap data...</div>
          </div>
        )}
        <ComposableMap
          projection="geoNaturalEarth1"
          projectionConfig={{
            scale: 210,
            center: [0, 0],
          }}
          style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }}
        >
          <ZoomableGroup center={[10, 15]} zoom={1.3} minZoom={1.3} maxZoom={1.3}>
            <Geographies geography={GEO_URL}>
              {({ geographies }) =>
                geographies
                  .filter((geo) => geo.properties.name !== "Antarctica")
                  .map((geo) => {
                  const countryName = geo.properties.name;
                  const alpha2 = NAME_TO_ALPHA2[countryName];
                  const isSelected = selectedCountry === alpha2;
                  const fillColor = getCountryColor(alpha2, isSelected);

                  return (
                    <Geography
                      key={geo.rsmKey}
                      geography={geo}
                      onMouseEnter={() => {
                        setHoveredCountry(countryName);
                        setHoveredCountryCode(alpha2);
                      }}
                      onMouseLeave={() => {
                        setHoveredCountry(null);
                        setHoveredCountryCode(null);
                      }}
                      onClick={() => {
                        if (alpha2) {
                          onCountrySelect(alpha2, countryName);
                        }
                      }}
                      style={{
                        default: {
                          fill: fillColor,
                          stroke: "#a1a1aa",
                          strokeWidth: 0.5,
                          outline: "none",
                          cursor: alpha2 ? "pointer" : "default",
                        },
                        hover: {
                          fill: isSelected ? "#16a34a" : alpha2 ? "#a3e635" : "#f4f4f5",
                          stroke: "#71717a",
                          strokeWidth: 0.75,
                          outline: "none",
                          cursor: alpha2 ? "pointer" : "default",
                        },
                        pressed: {
                          fill: "#16a34a",
                          stroke: "#52525b",
                          strokeWidth: 1,
                          outline: "none",
                        },
                      }}
                    />
                  );
                })
              }
            </Geographies>
          </ZoomableGroup>
        </ComposableMap>
      </div>
    </div>
  );
}

export default memo(WorldMap);
