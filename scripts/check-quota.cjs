/**
 * Check quota status for a user
 * Usage: node scripts/check-quota.js [ownerId]
 * 
 * If no ownerId is provided, checks quota for the default user: hPCczivUlYSJsSMAkr0EwEsG8z12
 */

const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const envPath = path.join(__dirname, '..', '.env');
const envContent = fs.readFileSync(envPath, 'utf8');
const lines = envContent.split('\n');

let databaseUrl = '';
for (const line of lines) {
  if (line.startsWith('DATABASE_URL=')) {
    databaseUrl = line.substring('DATABASE_URL='.length);
    break;
  }
}

if (!databaseUrl) {
  console.error('DATABASE_URL not found in .env');
  process.exit(1);
}

const DEFAULT_OWNER_ID = 'hPCczivUlYSJsSMAkr0EwEsG8z12';

async function checkQuota(ownerId = DEFAULT_OWNER_ID) {
  console.log(`Checking quota for user: ${ownerId}\n`);
  
  const client = new Client({
    connectionString: databaseUrl,
    ssl: false
  });

  try {
    await client.connect();
    
    const result = await client.query(
      'SELECT * FROM monthly_usage_counters WHERE owner_id = $1 ORDER BY year_month DESC',
      [ownerId]
    );
    
    if (result.rows.length === 0) {
      console.log('No quota records found for this user.');
      return;
    }
    
    console.log('Quota history:');
    console.log('================');
    
    for (const row of result.rows) {
      console.log(`\nMonth: ${row.year_month}`);
      console.log(`  mini_chat_used: ${row.mini_chat_used}`);
      console.log(`  pro_chat_used: ${row.pro_chat_used}`);
      console.log(`  kundli_generate_used: ${row.kundli_generate_used}`);
      const updatedAt = row.updated_at;
      if (updatedAt) {
        const date = new Date(Number(updatedAt) * 1000);
        console.log(`  updated_at: ${isNaN(date.getTime()) ? String(updatedAt) : date.toISOString()}`);
      } else {
        console.log(`  updated_at: N/A`);
      }
    }
    
    await client.end();
  } catch (error) {
    console.error('Error:', error.message);
    try { await client.end(); } catch (e) {}
    process.exit(1);
  }
}

// Get ownerId from command line argument
const ownerId = process.argv[2];
checkQuota(ownerId);