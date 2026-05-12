/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "assets.ngc.nvidia.com"
      }
    ]
  }
};

export default nextConfig;
