import 'dotenv/config';
import { Pool } from 'pg';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';
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

async function getFirebaseAuth() {
  if (getApps().length === 0) {
    const serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT_JSON?.trim();
    if (!serviceAccountJson) {
      throw new Error('FIREBASE_SERVICE_ACCOUNT_JSON env var is required');
    }
    initializeApp({
      credential: cert(JSON.parse(serviceAccountJson)),
      projectId: process.env.FIREBASE_PROJECT_ID,
    });
  }
  return getAuth();
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

async function getExistingTables(pool) {
  const result = await pool.query(
    `SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name`
  );
  return new Set(result.rows.map(r => r.table_name));
}

// Tables with owner_id column — delete order matters for FK constraints
const OWNER_ID_TABLES = [
  'chat_messages',
  'chat_sessions',
  'rag_profiles',
  'chart_jobs',
  'charts',
  'monthly_usage_counters',
  'subscriptions',
  'auth_users',
];

const DELETE_DOCUMENTS = `DELETE FROM documents WHERE collection IN ('rag_profiles', 'chat_sessions', 'chat_messages', 'auth_users') AND data->>'ownerId' = $1`;

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 1 && !args[0].startsWith('--')) {
    const maybe = args[0];
    if (/^[+()0-9\-\s]+$/.test(maybe)) {
      args.unshift('--phone');
      args[1] = maybe;
    } else {
      args.unshift('--owner-id');
      args[1] = maybe;
    }
  }

  if (hasFlag(args, '--help') || hasFlag(args, '-h')) {
    console.log([
      'Usage:',
      '  node scripts/delete-user.mjs --phone +19879879879',
      '  node scripts/delete-user.mjs --owner-id <ownerId>',
      '',
      'Options:',
      '  --phone <number>     Resolve the owner ID from auth_users.phone_number',
      '  --owner-id <id>      Delete by owner ID directly',
      '  --dry-run            Print what would be deleted without actually deleting',
      '  --skip-firebase      Skip deleting the Firebase Auth user',
    ].join('\n'));
    return;
  }

  const options = {
    phone: getArgValue(args, ['--phone']),
    ownerId: getArgValue(args, ['--owner-id']),
    dryRun: hasFlag(args, '--dry-run'),
    skipFirebase: hasFlag(args, '--skip-firebase'),
  };

  if (!options.phone && !options.ownerId) {
    throw new Error('Pass --phone or --owner-id. Use --help for examples.');
  }

  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is required');
  }

  const pool = new Pool({
    connectionString: withPostgresSslOverrides(databaseUrl),
    ...buildPostgresSslConfig(),
  });

  try {
    const ownerId = await resolveOwnerId(pool, options);
    const existingTables = await getExistingTables(pool);

    console.log(`\n=== Deleting all data for owner_id: ${ownerId} ===\n`);

    // Count rows per table
    const counts = [];

    for (const table of OWNER_ID_TABLES) {
      if (existingTables.has(table)) {
        const result = await pool.query(`SELECT COUNT(*) AS c FROM ${table} WHERE owner_id = $1`, [ownerId]);
        const count = parseInt(result.rows[0].c, 10);
        if (count > 0) counts.push({ table, count });
      }
    }

    // Documents table (JSONB store)
    if (existingTables.has('documents')) {
      const docResult = await pool.query(
        `SELECT COUNT(*) AS c FROM documents WHERE collection IN ('rag_profiles', 'chat_sessions', 'chat_messages', 'auth_users') AND data->>'ownerId' = $1`,
        [ownerId]
      );
      const docCount = parseInt(docResult.rows[0].c, 10);
      if (docCount > 0) counts.push({ table: 'documents', count: docCount });
    }

    if (counts.length === 0) {
      console.log('No records found for this user. Nothing to delete.\n');
      return;
    }

    console.log('Records to delete:');
    let total = 0;
    for (const { table, count } of counts) {
      console.log(`  ${table}: ${count}`);
      total += count;
    }
    console.log(`  TOTAL: ${total} records\n`);

    if (options.dryRun) {
      console.log('DRY RUN — no changes made.\n');
      return;
    }

    // Delete documents first (no FK constraints)
    const docEntry = counts.find(c => c.table === 'documents');
    if (docEntry) {
      const result = await pool.query(DELETE_DOCUMENTS, [ownerId]);
      console.log(`  Deleted ${result.rowCount} from documents`);
    }

    // Delete owner_id tables in order
    for (const table of OWNER_ID_TABLES) {
      const entry = counts.find(c => c.table === table);
      if (entry) {
        const result = await pool.query(`DELETE FROM ${table} WHERE owner_id = $1`, [ownerId]);
        console.log(`  Deleted ${result.rowCount} from ${table}`);
      }
    }

    console.log('\nPostgreSQL data deleted successfully.\n');

    // Delete from Firebase Auth
    if (!options.skipFirebase) {
      try {
        const auth = await getFirebaseAuth();
        await auth.deleteUser(ownerId);
        console.log('Firebase Auth user deleted successfully.\n');
      } catch (fbError) {
        console.warn('Warning: Failed to delete Firebase Auth user:', fbError.message);
        console.warn('The user may not exist in Firebase Auth, or the service account may not have sufficient permissions.\n');
      }
    } else {
      console.log('Skipped Firebase Auth deletion (--skip-firebase).\n');
    }

    console.log('=== User deletion complete ===\n');
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error('Failed to delete user:', error);
  process.exit(1);
});
