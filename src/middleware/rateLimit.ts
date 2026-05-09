import type { Request, Response, NextFunction } from 'express';
import { env } from '../config/env.js';

interface Bucket {
  count: number;
  resetAt: number;
}

const expensiveBuckets = new Map<string, Bucket>();

function getClientIp(req: Request): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded.trim()) {
    return forwarded.split(',')[0]!.trim();
  }
  return req.ip || 'unknown';
}

function compactBuckets(now: number) {
  if (expensiveBuckets.size < 1000) return;
  for (const [key, value] of expensiveBuckets) {
    if (value.resetAt <= now) expensiveBuckets.delete(key);
  }
}

export function expensiveEndpointRateLimit(req: Request, res: Response, next: NextFunction) {
  const now = Date.now();
  compactBuckets(now);

  const uid = req.user?.uid;
  const ip = getClientIp(req);
  const key = `${uid ?? 'anon'}:${ip}:${req.path}`;

  const existing = expensiveBuckets.get(key);
  if (!existing || existing.resetAt <= now) {
    expensiveBuckets.set(key, {
      count: 1,
      resetAt: now + env.RATE_LIMIT_CONFIG.window_ms,
    });
    res.setHeader('X-RateLimit-Limit', String(env.RATE_LIMIT_CONFIG.expensive_max));
    res.setHeader('X-RateLimit-Remaining', String(Math.max(0, env.RATE_LIMIT_CONFIG.expensive_max - 1)));
    return next();
  }

  existing.count += 1;
  expensiveBuckets.set(key, existing);

  const remaining = Math.max(0, env.RATE_LIMIT_CONFIG.expensive_max - existing.count);
  res.setHeader('X-RateLimit-Limit', String(env.RATE_LIMIT_CONFIG.expensive_max));
  res.setHeader('X-RateLimit-Remaining', String(remaining));

  if (existing.count > env.RATE_LIMIT_CONFIG.expensive_max) {
    const retryAfterSec = Math.max(1, Math.ceil((existing.resetAt - now) / 1000));
    res.setHeader('Retry-After', String(retryAfterSec));
    return res.status(429).json({
      error: 'Too many requests',
      details: `Expensive endpoint rate limit exceeded. Retry in ${retryAfterSec}s.`,
    });
  }

  return next();
}
