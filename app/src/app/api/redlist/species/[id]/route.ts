import { NextRequest, NextResponse } from "next/server";

interface CommonName {
  name: string;
  language?: string;
  main?: boolean;
}

interface IUCNAssessment {
  criteria: string | null;
  taxon?: {
    common_names?: CommonName[];
  };
}

interface IUCNTaxon {
  assessments?: { assessment_id: number }[];
  taxon?: {
    common_names?: CommonName[];
  };
}

interface INatTaxon {
  results?: {
    default_photo?: {
      square_url?: string;
      medium_url?: string;
      url?: string;
    };
  }[];
}

// Cache for species details (1 hour)
const detailsCache = new Map<string, { data: object; timestamp: number }>();
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour

async function fetchWithAuth(url: string): Promise<Response> {
  const apiKey = process.env.RED_LIST_API_KEY;
  if (!apiKey) {
    throw new Error("RED_LIST_API_KEY environment variable not set");
  }

  return fetch(url, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const searchParams = request.nextUrl.searchParams;
  const assessmentId = searchParams.get("assessmentId");
  const scientificName = searchParams.get("name");
  const assessmentYear = searchParams.get("assessmentYear"); // Year of last assessment for filtering GBIF data
  const assessmentMonth = searchParams.get("assessmentMonth"); // Month of last assessment (1-12) for more accurate filtering

  const speciesId = parseInt(id);
  const cacheKey = `${speciesId}-${assessmentYear || 'none'}-${assessmentMonth || 'none'}`;

  // Check cache
  const cached = detailsCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return NextResponse.json({ ...cached.data, cached: true });
  }

  try {
    // Fetch IUCN taxon details (for assessment count and common name) and GBIF data in parallel
    const promises: Promise<Response>[] = [
      // IUCN taxon details (for assessment count and common name)
      fetchWithAuth(`https://api.iucnredlist.org/api/v4/taxa/sis/${speciesId}`),
      // GBIF species search by name (for GBIF link and taxon key)
      ...(scientificName ? [fetch(`https://api.gbif.org/v1/species/match?name=${encodeURIComponent(scientificName)}`)] : []),
    ];

    // Also fetch assessment details for criteria if we have assessmentId
    if (assessmentId) {
      promises.push(
        fetchWithAuth(`https://api.iucnredlist.org/api/v4/assessment/${assessmentId}`)
      );
    }

    const responses = await Promise.all(promises);

    let criteria: string | null = null;
    let commonName: string | null = null;
    let gbifUrl: string | null = null;
    let gbifOccurrences: number | null = null;
    let assessmentCount = 1;
    let inatDefaultImage: { squareUrl: string | null; mediumUrl: string | null } | null = null;

    // Parse IUCN taxon response (for assessment count and common name)
    if (responses[0]?.ok) {
      const taxonData: IUCNTaxon = await responses[0].json();
      assessmentCount = taxonData.assessments?.length || 1;

      // Get common name from taxon data
      if (taxonData.taxon?.common_names && taxonData.taxon.common_names.length > 0) {
        const names = taxonData.taxon.common_names;
        const englishName = names.find((n) => n.language === "eng" || n.language === "en");
        const mainName = names.find((n) => n.main);
        commonName = englishName?.name || mainName?.name || names[0]?.name || null;
      }
    }

    // Parse GBIF response and fetch occurrence counts
    let gbifOccurrencesSinceAssessment: number | null = null;
    let gbifByRecordType: { humanObservation: number; preservedSpecimen: number; machineObservation: number; other: number; iNaturalist: number } | null = null;
    let gbifNewByRecordType: { humanObservation: number; preservedSpecimen: number; machineObservation: number; other: number; iNaturalist: number } | null = null;
    let recentInatObservations: { url: string; date: string | null; imageUrl: string | null; location: string | null; observer: string | null }[] = [];
    let inatTotalCount = 0;

    const gbifIndex = 1;
    if (scientificName && responses[gbifIndex]?.ok) {
      const gbifMatch = await responses[gbifIndex].json();
      if (gbifMatch.usageKey) {
        gbifUrl = `https://www.gbif.org/species/${gbifMatch.usageKey}`;
        const taxonKey = gbifMatch.usageKey;

        // Fetch all GBIF occurrence counts in parallel
        try {
          const currentYear = new Date().getFullYear();
          // iNaturalist dataset key in GBIF
          const INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7";

          const gbifPromises: Promise<Response>[] = [
            // Index 0: Total occurrences
            fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}`),
            // Index 1-3: By record type
            fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&basisOfRecord=HUMAN_OBSERVATION`),
            fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&basisOfRecord=PRESERVED_SPECIMEN`),
            fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&basisOfRecord=MACHINE_OBSERVATION`),
          ];

          // Fetch iNaturalist count and recent observations (up to 5 for navigation)
          const inatCountPromise = fetch(
            `https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&datasetKey=${INAT_DATASET_KEY}`
          );
          const inatRecentPromise = fetch(
            `https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&datasetKey=${INAT_DATASET_KEY}&limit=5`
          );

          // Add since-assessment queries if we have assessment year
          // Use search endpoint with limit=0 for month-accurate filtering
          let inatNewCountPromise: Promise<Response> | null = null;
          let sameYearPromises: Promise<Response>[] = [];
          const parsedYear = assessmentYear ? parseInt(assessmentYear) : null;
          const parsedMonth = assessmentMonth ? parseInt(assessmentMonth) : null;

          if (parsedYear) {
            const startYear = parsedYear + 1; // Full years after assessment
            const yearRange = startYear <= currentYear ? `${startYear},${currentYear}` : null;

            // For the assessment year itself, get occurrences from months AFTER the assessment month
            // Using search endpoint with limit=0 to get count with month filtering
            if (parsedMonth && parsedMonth < 12) {
              const monthRange = `${parsedMonth + 1},12`; // Months after assessment in same year
              sameYearPromises = [
                // Same year, later months - total
                fetch(`https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&year=${parsedYear}&month=${monthRange}&limit=0`),
                // Same year, later months - by record type
                fetch(`https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&year=${parsedYear}&month=${monthRange}&basisOfRecord=HUMAN_OBSERVATION&limit=0`),
                fetch(`https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&year=${parsedYear}&month=${monthRange}&basisOfRecord=PRESERVED_SPECIMEN&limit=0`),
                fetch(`https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&year=${parsedYear}&month=${monthRange}&basisOfRecord=MACHINE_OBSERVATION&limit=0`),
                // Same year, later months - iNaturalist
                fetch(`https://api.gbif.org/v1/occurrence/search?taxonKey=${taxonKey}&year=${parsedYear}&month=${monthRange}&datasetKey=${INAT_DATASET_KEY}&limit=0`),
              ];
            }

            if (yearRange) {
              gbifPromises.push(
                // Index 4: Total new occurrences (full years after)
                fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&year=${yearRange}`),
                // Index 5-7: New occurrences by record type (full years after)
                fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&year=${yearRange}&basisOfRecord=HUMAN_OBSERVATION`),
                fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&year=${yearRange}&basisOfRecord=PRESERVED_SPECIMEN`),
                fetch(`https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&year=${yearRange}&basisOfRecord=MACHINE_OBSERVATION`)
              );
              // iNaturalist count for full years after
              inatNewCountPromise = fetch(
                `https://api.gbif.org/v1/occurrence/count?taxonKey=${taxonKey}&datasetKey=${INAT_DATASET_KEY}&year=${yearRange}`
              );
            }
          }

          // Fetch iNaturalist default species image
          const inatTaxaPromise = fetch(
            `https://api.inaturalist.org/v1/taxa?q=${encodeURIComponent(scientificName)}&rank=species&per_page=1`
          );

          const [gbifResponses, inatCountResponse, inatRecentResponse, inatNewCountResponse, sameYearResponses, inatTaxaResponse] = await Promise.all([
            Promise.all(gbifPromises),
            inatCountPromise,
            inatRecentPromise,
            inatNewCountPromise || Promise.resolve(null),
            sameYearPromises.length > 0 ? Promise.all(sameYearPromises) : Promise.resolve([]),
            inatTaxaPromise,
          ]);

          // Parse responses
          if (gbifResponses[0]?.ok) {
            gbifOccurrences = await gbifResponses[0].json();
          }

          // Parse iNaturalist count and recent observations
          if (inatCountResponse?.ok) {
            inatTotalCount = await inatCountResponse.json();
          }

          // Parse iNaturalist default species image
          if (inatTaxaResponse?.ok) {
            const inatTaxaData: INatTaxon = await inatTaxaResponse.json();
            const defaultPhoto = inatTaxaData.results?.[0]?.default_photo;
            if (defaultPhoto) {
              inatDefaultImage = {
                squareUrl: defaultPhoto.square_url || defaultPhoto.url || null,
                mediumUrl: defaultPhoto.medium_url || defaultPhoto.url || null,
              };
            }
          }
          if (inatRecentResponse?.ok) {
            const inatData = await inatRecentResponse.json();
            if (inatData.results && inatData.results.length > 0) {
              recentInatObservations = inatData.results
                .filter((obs: { references?: string }) => obs.references)
                .map((obs: { references: string; eventDate?: string; media?: { identifier?: string }[]; verbatimLocality?: string; stateProvince?: string; country?: string; recordedBy?: string }) => {
                  const imageUrl = obs.media?.[0]?.identifier || null;
                  const locationParts = [obs.verbatimLocality, obs.stateProvince, obs.country].filter(Boolean);
                  const location = locationParts.length > 0 ? locationParts.join(', ') : null;
                  return {
                    url: obs.references,
                    date: obs.eventDate ? obs.eventDate.split('T')[0] : null,
                    imageUrl,
                    location,
                    observer: obs.recordedBy || null,
                  };
                });
            }
          }

          // Parse record type breakdown
          const [, humanResp, specimenResp, machineResp] = gbifResponses;
          const humanObs = humanResp?.ok ? await humanResp.json() : 0;
          const specimen = specimenResp?.ok ? await specimenResp.json() : 0;
          const machine = machineResp?.ok ? await machineResp.json() : 0;
          const other = (gbifOccurrences || 0) - humanObs - specimen - machine;

          gbifByRecordType = {
            humanObservation: humanObs,
            preservedSpecimen: specimen,
            machineObservation: machine,
            other: Math.max(0, other),
            iNaturalist: inatTotalCount,
          };

          // Parse since-assessment counts if we requested them
          // Helper to extract count from search response (returns {count: N})
          const getSearchCount = async (resp: Response | undefined): Promise<number> => {
            if (!resp?.ok) return 0;
            const data = await resp.json();
            return data.count || 0;
          };

          if (parsedYear) {
            // Parse same-year counts (from search endpoint with month filtering)
            let sameYearTotal = 0, sameYearHuman = 0, sameYearSpecimen = 0, sameYearMachine = 0, sameYearInat = 0;
            if (sameYearResponses.length > 0) {
              [sameYearTotal, sameYearHuman, sameYearSpecimen, sameYearMachine, sameYearInat] = await Promise.all([
                getSearchCount(sameYearResponses[0]),
                getSearchCount(sameYearResponses[1]),
                getSearchCount(sameYearResponses[2]),
                getSearchCount(sameYearResponses[3]),
                getSearchCount(sameYearResponses[4]),
              ]);
            }

            // Parse full-years-after counts (from count endpoint)
            let afterYearsTotal = 0, afterYearsHuman = 0, afterYearsSpecimen = 0, afterYearsMachine = 0, afterYearsInat = 0;
            if (gbifResponses[4]?.ok) {
              afterYearsTotal = await gbifResponses[4].json();
              afterYearsHuman = gbifResponses[5]?.ok ? await gbifResponses[5].json() : 0;
              afterYearsSpecimen = gbifResponses[6]?.ok ? await gbifResponses[6].json() : 0;
              afterYearsMachine = gbifResponses[7]?.ok ? await gbifResponses[7].json() : 0;
              afterYearsInat = inatNewCountResponse?.ok ? await inatNewCountResponse.json() : 0;
            }

            // Combine same-year + after-years counts
            const totalNew = sameYearTotal + afterYearsTotal;
            const newHumanObs = sameYearHuman + afterYearsHuman;
            const newSpecimen = sameYearSpecimen + afterYearsSpecimen;
            const newMachine = sameYearMachine + afterYearsMachine;
            const newInat = sameYearInat + afterYearsInat;
            const newOther = totalNew - newHumanObs - newSpecimen - newMachine;

            gbifOccurrencesSinceAssessment = totalNew;
            gbifNewByRecordType = {
              humanObservation: newHumanObs,
              preservedSpecimen: newSpecimen,
              machineObservation: newMachine,
              other: Math.max(0, newOther),
              iNaturalist: newInat,
            };
          }
        } catch {
          // Ignore occurrence fetch errors
        }
      }
    }

    // Parse assessment details for criteria
    const assessmentIndex = scientificName ? 2 : 1;
    if (assessmentId && responses[assessmentIndex]?.ok) {
      const assessmentData: IUCNAssessment = await responses[assessmentIndex].json();
      criteria = assessmentData.criteria;
    }

    const result = {
      sis_taxon_id: speciesId,
      criteria,
      commonName,
      gbifUrl,
      gbifOccurrences,
      gbifOccurrencesSinceAssessment,
      gbifByRecordType,
      gbifNewByRecordType,
      recentInatObservations,
      inatTotalCount,
      inatDefaultImage,
      assessmentCount,
    };

    // Cache the result
    detailsCache.set(cacheKey, { data: result, timestamp: Date.now() });

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error fetching species details:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
