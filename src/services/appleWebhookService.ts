import fs from 'fs';
import path from 'path';
import { SignedDataVerifier, VerificationException, VerificationStatus, Environment } from '@apple/app-store-server-library';
import type { ResponseBodyV2DecodedPayload } from '@apple/app-store-server-library';
import type { JWSTransactionDecodedPayload } from '@apple/app-store-server-library';
import { env } from '../config/env.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { getPostgresPool } from './postgresClient.js';

const WEBHOOK_LOG_PATH = path.resolve(process.cwd(), 'webhook-log.txt');

export function appendLog(entry: string): void {
  try {
    fs.appendFileSync(WEBHOOK_LOG_PATH, `${entry}\n`);
  } catch {
    // non-fatal logging error
  }
}

function logEvent(
  direction: 'WEBHOOK' | 'SYNC',
  eventType: string,
  status: string,
  details: Record<string, string | undefined>,
  error?: string,
): void {
  const timestamp = new Date().toISOString();
  const parts = [
    timestamp,
    direction,
    eventType,
    status,
    ...Object.entries(details).map(([k, v]) => `${k}=${v ?? '-'}`),
    error ? `error=${error}` : '',
  ].filter(Boolean);
  appendLog(parts.join('  '));
}

const APPLE_ROOT_CA_B64 = 'MIIEuzCCA6OgAwIBAgIBAjANBgkqhkiG9w0BAQUFADBiMQswCQYDVQQGEwJVUzETMBEGA1UEChMKQXBwbGUgSW5jLjEmMCQGA1UECxMdQXBwbGUgQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkxFjAUBgNVBAMTDUFwcGxlIFJvb3QgQ0EwHhcNMDYwNDI1MjE0MDM2WhcNMzUwMjA5MjE0MDM2WjBiMQswCQYDVQQGEwJVUzETMBEGA1UEChMKQXBwbGUgSW5jLjEmMCQGA1UECxMdQXBwbGUgQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkxFjAUBgNVBAMTDUFwcGxlIFJvb3QgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDkkakJH5HbHkdQ6wXtXnmELes2oldMVeyLGYne+Uts9QerIjAC6Bg++FAJ039BqJj50cpmnCRrEdCju+QbKsMflZ56DKRHi1vUFjczy8QPTc4UadHJGXL1XQ7Vf1+b8iUDulWPTV0N8WQ1IxVLFVkds5T39pyez1C6wVhQZ48ItCD3y6wsIG9wtj8BMIy3Q88PnT3zK0koGsj+zrW5DtleHNbLPbU6rfQPDgCSC7EhFi501TwN22IWq6NxkkdTVcGvL0Gz+PvjcM3mo0xFfh9Ma1CWQYnEdGILEINBhzOKgbEwWOxaBDKMaLOPHd5lc/9nXmW8Sdh2nzMUZaF3lMktAgMBAAGjggF6MIIBdjAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUK9BpR5R2Cf70a40uQKb3R01/CF4wHwYDVR0jBBgwFoAUK9BpR5R2Cf70a40uQKb3R01/CF4wggERBgNVHSAEggEIMIIBBDCCAQAGCSqGSIb3Y2QFATCB8jAqBggrBgEFBQcCARYeaHR0cHM6Ly93d3cuYXBwbGUuY29tL2FwcGxlY2EvMIHDBggrBgEFBQcCAjCBthqBs1JlbGlhbmNlIG9uIHRoaXMgY2VydGlmaWNhdGUgYnkgYW55IHBhcnR5IGFzc3VtZXMgYWNjZXB0YW5jZSBvZiB0aGUgdGhlbiBhcHBsaWNhYmxlIHN0YW5kYXJkIHRlcm1zIGFuZCBjb25kaXRpb25zIG9mIHVzZSwgY2VydGlmaWNhdGUgcG9saWN5IGFuZCBjZXJ0aWZpY2F0aW9uIHByYWN0aWNlIHN0YXRlbWVudHMuMA0GCSqGSIb3DQEBBQUAA4IBAQBcNplMLXi37Yyb3PN3m/J20ncwT8EfhYOFG5k9RzfyqZtAjizUsZAS2L70c5vu0mQPy3lPNNiiPvl4/2vIB+x9OYOLUyDTOMSxv5pPCmv/K/xZpwUJfBdAVhEedNO3iyM7R6PVbyTi69G3cN8PReEnyvFteO3ntRcXqNx+IjXKJdXZD9Zr1KIkIxH3oayPc4FgxhtbCS+SsvhESPBgOJ4V9T0mZyCKM2r3DYLP3uujL/lTaltkwGMzd/c6ByxW69oPIQ7aunMZT7XZNn/Bh1XZp5m5MkL72NVxnn6hUrcbvZNCJBIqxw8dtk2cXmPIS4AXUKqK1drk/NAJBzewdXUh';

const rootCertBuffer = Buffer.from(APPLE_ROOT_CA_B64, 'base64');

let productionVerifier: SignedDataVerifier | null = null;
let sandboxVerifier: SignedDataVerifier | null = null;

function getProductionVerifier(): SignedDataVerifier {
  if (!productionVerifier) {
    const bundleId = env.APPLE_BUNDLE_ID;
    if (!bundleId) {
      throw new Error('APPLE_BUNDLE_ID is required for Apple webhook verification');
    }
    productionVerifier = new SignedDataVerifier(
      [rootCertBuffer],
      true,
      Environment.PRODUCTION,
      bundleId,
      env.APPLE_APP_ID ? Number(env.APPLE_APP_ID) : undefined,
    );
  }
  return productionVerifier;
}

function getSandboxVerifier(): SignedDataVerifier {
  if (!sandboxVerifier) {
    const bundleId = env.APPLE_BUNDLE_ID;
    if (!bundleId) {
      throw new Error('APPLE_BUNDLE_ID is required for Apple webhook verification');
    }
    sandboxVerifier = new SignedDataVerifier(
      [rootCertBuffer],
      true,
      Environment.SANDBOX,
      bundleId,
    );
  }
  return sandboxVerifier;
}

async function verifyAndDecodePayload(signedPayload: string): Promise<ResponseBodyV2DecodedPayload> {
  try {
    return await getProductionVerifier().verifyAndDecodeNotification(signedPayload);
  } catch (error) {
    if (error instanceof VerificationException && error.status === VerificationStatus.INVALID_ENVIRONMENT) {
      try {
        return await getSandboxVerifier().verifyAndDecodeNotification(signedPayload);
      } catch (sandboxError) {
        logEvent('WEBHOOK', 'VERIFY', 'sandbox_fail', {}, String(sandboxError));
        throw sandboxError;
      }
    }
    throw error;
  }
}

const processedNotifications = new Set<string>();

function decodeTransaction(signedInfo: string): JWSTransactionDecodedPayload {
  const parts = signedInfo.split('.');
  if (parts.length !== 3) {
    throw new Error('Invalid JWS: expected 3 parts');
  }
  return JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf-8'));
}

export interface WebhookResult {
  handled: boolean;
  eventType?: string;
  ownerId?: string;
  error?: string;
}

export async function processAppleWebhook(body: { signedPayload?: string }): Promise<WebhookResult> {
  if (!body.signedPayload || typeof body.signedPayload !== 'string') {
    return { handled: false, error: 'missing signedPayload' };
  }

  let decoded: ResponseBodyV2DecodedPayload;

  try {
    decoded = await verifyAndDecodePayload(body.signedPayload);
  } catch (error) {
    if (error instanceof VerificationException) {
      return { handled: false, error: `verification_failed: ${VerificationStatus[error.status] ?? error.status}` };
    }
    return { handled: false, error: `verification_error: ${String(error)}` };
  }

  if (!decoded.notificationType) {
    return { handled: false, error: 'missing notificationType' };
  }

  if (decoded.notificationUUID) {
    if (processedNotifications.has(decoded.notificationUUID)) {
      logEvent('WEBHOOK', decoded.notificationType as string, 'duplicate_skipped', { uuid: decoded.notificationUUID });
      return { handled: true, eventType: decoded.notificationType as string, ownerId: 'duplicate_skipped' };
    }
    processedNotifications.add(decoded.notificationUUID);
    if (processedNotifications.size > 10000) {
      processedNotifications.clear();
    }
  }

  if (decoded.notificationType === 'TEST') {
    logEvent('WEBHOOK', 'TEST', 'ok', {});
    return { handled: true, eventType: 'TEST' };
  }

  if (!decoded.data?.signedTransactionInfo) {
    if (decoded.notificationType === 'RENEWAL_EXTENSION') {
      logEvent('WEBHOOK', decoded.notificationType as string, 'ok', { detail: 'no_transaction_info' });
      return { handled: true, eventType: 'RENEWAL_EXTENSION' };
    }
    logEvent('WEBHOOK', decoded.notificationType as string, 'ignored', {}, 'missing signedTransactionInfo');
    return { handled: false, error: 'missing signedTransactionInfo' };
  }

  let tx: JWSTransactionDecodedPayload;
  try {
    tx = decodeTransaction(decoded.data.signedTransactionInfo);
  } catch {
    logEvent('WEBHOOK', decoded.notificationType as string, 'error', {}, 'invalid signedTransactionInfo JWT');
    return { handled: false, error: 'invalid signedTransactionInfo JWT' };
  }

  const originalTransactionId = tx.originalTransactionId;
  if (!originalTransactionId) {
    return { handled: false, error: 'missing originalTransactionId in transaction' };
  }

  const subscriptions = getSubscriptionsRepository();
  const sub = await subscriptions.getByOriginalTransactionId(originalTransactionId);

  if (!sub) {
    const pool = getPostgresPool();

    if (pool) {
      try {
        await pool.query(
          `INSERT INTO apple_webhook_orphans (original_transaction_id, notification_type, product_id, expires_date, received_at)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (original_transaction_id) DO UPDATE SET
             notification_type = EXCLUDED.notification_type,
             received_at = EXCLUDED.received_at`,
          [
            originalTransactionId,
            decoded.notificationType as string,
            tx.productId ?? null,
            tx.expiresDate ?? null,
            Date.now(),
          ]
        );
      } catch {
        // non-fatal: orphan logging is best-effort
      }
    }

    logEvent('WEBHOOK', decoded.notificationType as string, 'orphaned', {
      originalTransactionId,
      productId: tx.productId ?? '-',
    }, 'no matching subscription');
    return { handled: false, error: `unmatched originalTransactionId: ${originalTransactionId}` };
  }

  switch (decoded.notificationType) {
    case 'SUBSCRIBED':
    case 'DID_RENEW':
    case 'OFFER_REDEEMED':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: true,
        eventType: decoded.notificationType === 'SUBSCRIBED' ? 'apple_subscribed' : 'apple_did_renew',
        expiresAtMs: tx.expiresDate,
        transactionId: tx.transactionId,
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'true',
        expiresAtMs: String(tx.expiresDate ?? '-'),
      });
      break;

    case 'EXPIRED':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: false,
        eventType: 'apple_expired',
        expiresAtMs: tx.expiresDate,
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'false',
      });
      break;

    case 'REFUND':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: false,
        eventType: 'apple_refund',
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'false',
      });
      break;

    case 'REFUND_REVERSED':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: true,
        eventType: 'apple_refund_reversed',
        expiresAtMs: tx.expiresDate,
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'true',
      });
      break;

    case 'GRACE_PERIOD_EXPIRED':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: false,
        eventType: 'apple_grace_period_expired',
        expiresAtMs: tx.expiresDate,
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'false',
      });
      break;

    case 'REVOKE':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: false,
        eventType: 'apple_revoke',
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'applied', {
        ownerId: sub.ownerId,
        isPro: 'false',
      });
      break;

    case 'DID_FAIL_TO_RENEW':
      await subscriptions.updateFromAppleWebhook(sub.ownerId, {
        isPro: true,
        eventType: 'apple_billing_issue',
      });
      logEvent('WEBHOOK', decoded.notificationType as string, 'logged', {
        ownerId: sub.ownerId,
        detail: 'billing_issue_grace_period',
      });
      break;

    case 'DID_CHANGE_RENEWAL_STATUS':
    case 'DID_CHANGE_RENEWAL_PREF':
    case 'PRICE_INCREASE':
      logEvent('WEBHOOK', decoded.notificationType as string, 'info_only', {
        ownerId: sub.ownerId,
      });
      break;

    default:
      logEvent('WEBHOOK', decoded.notificationType as string, 'unhandled', {
        ownerId: sub.ownerId ?? '-',
      });
      break;
  }

  return {
    handled: true,
    eventType: decoded.notificationType as string,
    ownerId: sub.ownerId,
  };
}
