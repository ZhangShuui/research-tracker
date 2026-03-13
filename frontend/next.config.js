/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["remark-gfm", "react-markdown", "remark-math", "rehype-katex"],
};

module.exports = nextConfig;
