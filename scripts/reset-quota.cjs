/**
 * Reset quota for a user in the monthly_usage_counters table
 * Usage: node scripts/reset-quota.js [ownerId]
 * 
 * If no ownerId is provided, resets quota for the default user: hPCczivUlYSJsSMAkr0EwEsG8z12
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

async function resetQuota(ownerId = DEFAULT_OWNER_ID) {
  console.log(`Resetting quota for user: ${ownerId}`);
  
  const client = new Client({
    connectionString: databaseUrl,
    ssl: true
  });

  try {
    await client.connect();
    console.log('Connected to database');
    
    // Reset all quotas
    const result = await client.query(
      'UPDATE monthly_usage_counters SET mini_chat_used = 0, pro_chat_used = 0, kundli_generate_used = 0 WHERE owner_id = $1',
      [ownerId]
    );
    
    console.log(`Quota reset successful. Rows updated: ${result.rowCount}`);
    
    // Show current quota status
    const statusResult = await client.query(
      'SELECT * FROM monthly_usage_counters WHERE owner_id = $1',
      [ownerId]
    );
    
    if (statusResult.rows.length > 0) {
      const row = statusResult.rows[0];
      console.log('\nCurrent quota status:');
      console.log(`  mini_chat_used: ${row.mini_chat_used}`);
      console.log(`  pro_chat_used: ${row.pro_chat_used}`);
      console.log(`  kundli_generate_used: ${row.kundli_generate_used}`);
      console.log(`  year_month: ${row.year_month}`);
    }
    
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    await client.end();
  }
}

// Get ownerId from command line argument
const ownerId = process.argv[2];
resetQuota(ownerId);