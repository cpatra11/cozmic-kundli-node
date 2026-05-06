import 'dotenv/config';
import { Pool } from 'pg';
import { buildPostgresSslConfig, withPostgresSslOverrides } from '../dist/services/postgresSsl.js';

function getArgValue(args, names) {
  for (let index = 0; index < args.length; index += 1) {
    const current = args[index];
    if (names.includes(current)) {
      const next = args[index + 1];
      if (next && !next.startsWith('--')) {
        return next;
      }
    }

    for (const name of names) {
      if (current.startsWith(`${name}=`)) {
        return current.slice(name.length + 1);
      }
    }
  }

  return undefined;
}

function hasFlag(args, flag) {
  return args.includes(flag);
}

function sqlString(value) {
  return `'${String(value).replace(/'/g, "''")}'`;
}

async function resolveOwnerId(pool, options) {
  if (options.ownerId) {
    return options.ownerId.trim();
  }

  if (!options.phone) {
    throw new Error('Provide either --owner-id or --phone.');
  }

  const phone = options.phone.trim();
  const result = await pool.query(
    'SELECT owner_id, phone_number FROM auth_users WHERE phone_number = $1 LIMIT 1',
    [phone]
  );

  const row = result.rows[0];
  if (!row?.owner_id) {
    throw new Error(`No auth_users row found for phone number ${phone}`);
  }

  return row.owner_id;
}

async function main() {
  const args = process.argv.slice(2);
  // Convenience: allow a single positional argument to be treated as --phone or --owner-id
  // If a single non-flag argument is provided, infer its type heuristically.
  if (args.length === 1 && !args[0].startsWith('--')) {
    const maybe = args[0];
    // crude heuristic: if it starts with '+' or digits and contains only common phone chars, treat as phone
    if (/^[+()0-9\-\s]+$/.test(maybe)) {
      args.unshift('--phone');
      args[1] = maybe;
    } else {
      // Fallback: treat as owner-id
      args.unshift('--owner-id');
      args[1] = maybe;
    }
  }
  if (hasFlag(args, '--help') || hasFlag(args, '-h')) {
    console.log([
      'Usage:',
      '  node scripts/set-subscription-free.mjs --phone +916371757105',
      '  node scripts/set-subscription-free.mjs --owner-id hPCczivUlYSJsSMAkr0EwEsG8z12',
      '',
      'Options:',
      '  --phone <number>     Resolve the owner ID from auth_users.phone_number',
      '  --owner-id <id>      Update a subscription row directly by owner ID',
      '  --dry-run            Print the current row and exit without updating',
    ].join('\n'));
    return;
  }

  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is required');
  }

  const options = {
    phone: getArgValue(args, ['--phone']),
    ownerId: getArgValue(args, ['--owner-id']),
    dryRun: hasFlag(args, '--dry-run'),
  };

  if (!options.phone && !options.ownerId) {
    throw new Error('Pass --phone or --owner-id. Use --help for examples.');
  }

  const pool = new Pool({
    connectionString: withPostgresSslOverrides(databaseUrl),
    ...buildPostgresSslConfig(),
  });

  try {
    const ownerId = await resolveOwnerId(pool, options);
    const before = await pool.query(
      `
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, expires_at_ms, updated_at, last_event_at, last_event_id
      FROM subscriptions
      WHERE owner_id = $1
      LIMIT 1
      `,
      [ownerId]
    );

    const beforeRow = before.rows[0] ?? null;
    if (!beforeRow) {
      throw new Error(`No subscriptions row found for owner ID ${ownerId}`);
    }

    const now = Date.now();

    if (options.dryRun) {
      console.log(
        JSON.stringify(
          {
            dryRun: true,
            ownerId,
            row: beforeRow,
            wouldUpdateTo: {
              entitlement_id: 'free',
              is_pro: false,
              event_type: 'admin_revoke',
              expires_at_ms: now - 1000,
              updated_at: now,
              last_event_at: now,
            },
          },
          null,
          2
        )
      );
      return;
    }

    const update = await pool.query(
      `
      UPDATE subscriptions
      SET entitlement_id = $1,
          is_pro = $2,
          event_type = $3,
          expires_at_ms = $4,
          updated_at = $5,
          last_event_at = $6
      WHERE owner_id = $7
      RETURNING owner_id, source, entitlement_id, is_pro, store, product_id, event_type, expires_at_ms, updated_at, last_event_at, last_event_id
      `,
      ['free', false, 'admin_revoke', now - 1000, now, now, ownerId]
    );

    const after = await pool.query(
      `
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, expires_at_ms, updated_at, last_event_at, last_event_id
      FROM subscriptions
      WHERE owner_id = $1
      LIMIT 1
      `,
      [ownerId]
    );

    console.log(
      JSON.stringify(
        {
          ownerId,
          phone: options.phone ?? null,
          before: beforeRow,
          after: after.rows[0] ?? null,
          update: update.rows[0] ?? null,
        },
        null,
        2
      )
    );
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error('❌ Failed to set subscription free:', error);
  process.exit(1);
});
