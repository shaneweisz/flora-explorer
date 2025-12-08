# Flora Explorer

A web application for exploring plant biodiversity data, combining GBIF occurrence data with IUCN Red List conservation status.

## Features

- **GBIF Explorer**: Search and browse plant species with occurrence data, images, and distribution maps
- **Red List Stats**: View IUCN Red List statistics for plant species including:
  - Distribution by conservation category
  - Assessment recency analysis
  - Species browser with filtering, sorting, and pagination

## Getting Started

```bash
cd app
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

Create `app/.env.local` with:

```
RED_LIST_API_KEY=your_iucn_api_key
```

## Tech Stack

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- Recharts
