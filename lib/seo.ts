export const SEO_CONFIG = {
  title: "사상고 생기부AI - 학생 생활기록부 AI작성 탐지기",
  description:
    "학생 생활기록부를 AI가 분석하고 완벽하게 업그레이드합니다. 오류 탐지, 강점 분석, 개선 제안을 받아보세요.",
  keywords: ["생기부", "생활기록부", "AI 분석", "학생부", "대학 입시", "생기부 오류", "생기부 분석", "사상고"],
  ogImage: "/og-image.png",
  twitterCard: "summary_large_image",
  url: "https://saenggibu-ai.vercel.app",
}

export function generateMetaTags(page?: {
  title?: string
  description?: string
  path?: string
}) {
  const title = page?.title ? `${page.title} | ${SEO_CONFIG.title}` : SEO_CONFIG.title
  const description = page?.description || SEO_CONFIG.description
  const url = page?.path ? `${SEO_CONFIG.url}${page.path}` : SEO_CONFIG.url

  return {
    title,
    description,
    keywords: SEO_CONFIG.keywords.join(", "),
    openGraph: {
      title,
      description,
      url,
      siteName: SEO_CONFIG.title,
      images: [
        {
          url: SEO_CONFIG.ogImage,
          width: 1200,
          height: 630,
          alt: title,
        },
      ],
      locale: "ko_KR",
      type: "website",
    },
    twitter: {
      card: SEO_CONFIG.twitterCard,
      title,
      description,
      images: [SEO_CONFIG.ogImage],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        "max-video-preview": -1,
        "max-image-preview": "large",
        "max-snippet": -1,
      },
    },
  }
}
