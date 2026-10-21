
const pool = require('./db');

async function testConnection() {
    try {
        const result = await pool.query(
            'SELECT current_user, current_database()'
        );

        console.log('PostgreSQL connection successful!');
        console.log('User:', result.rows[0].current_user);
        console.log('Database:', result.rows[0].current_database);
    } catch (error) {
        console.error('Connection failed:', error.message);
    } finally {
        await pool.end();
    }
}

testConnection();
