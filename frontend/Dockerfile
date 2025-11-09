# ---------------------------
# 1️⃣ Build stage
# ---------------------------
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first for caching
COPY package.json package-lock.json* ./

# Install dependencies
RUN if [ -f package-lock.json ]; then npm ci --omit=dev; \
    else echo "Warning: package-lock.json not found" && npm install --omit=dev; fi

# Copy the rest of the app
COPY . .

# Build the React app
RUN npm run build

# ---------------------------
# 2️⃣ Production stage
# ---------------------------
FROM nginx:alpine

# Remove default Nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy built React app
COPY --from=builder /app/build /usr/share/nginx/html

# Copy your custom Nginx server block
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create /tmp for PID and ensure permissions
RUN mkdir -p /tmp && \
    chown -R nginx:nginx /usr/share/nginx/html /var/cache/nginx /var/log/nginx /etc/nginx/conf.d

# Use /tmp for PID to avoid permission errors
RUN sed -i 's#/run/nginx.pid#/tmp/nginx.pid#' /etc/nginx/nginx.conf

# Run as non-root
USER nginx

# Expose HTTP port
EXPOSE 80

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
