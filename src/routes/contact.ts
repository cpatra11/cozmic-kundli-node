import { Router } from 'express';
import nodemailer from 'nodemailer';
import { env } from '../config/env.js';

const router = Router();

function createTransporter() {
  if (!env.CONTACT_SMTP_USER || !env.CONTACT_SMTP_PASS) {
    return null;
  }
  return nodemailer.createTransport({
    host: 'smtp.gmail.com',
    port: 465,
    secure: true,
    auth: {
      user: env.CONTACT_SMTP_USER,
      pass: env.CONTACT_SMTP_PASS,
    },
  });
}

router.post('/v1/contact', async (req, res) => {
  try {
    const transporter = createTransporter();
    if (!transporter) {
      return res.status(500).json({ error: 'Contact form is not configured yet.' });
    }

    const { name, email, subject, message } = req.body;

    if (!name?.trim() || !email?.trim() || !message?.trim()) {
      return res.status(400).json({ error: 'Name, email, and message are required.' });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      return res.status(400).json({ error: 'Invalid email address.' });
    }

    if (message.trim().length < 10) {
      return res.status(400).json({ error: 'Message must be at least 10 characters.' });
    }

    await transporter.sendMail({
      from: env.CONTACT_SMTP_USER,
      to: env.CONTACT_EMAIL_TO,
      replyTo: email.trim(),
      subject: `[Cozmic Contact] ${subject?.trim() || 'No subject'} — from ${name.trim()}`,
      text: `Name: ${name.trim()}\nEmail: ${email.trim()}\n\nMessage:\n${message.trim()}`,
      html: `
        <h3>New Contact Form Submission</h3>
        <p><strong>Name:</strong> ${name.trim()}</p>
        <p><strong>Email:</strong> ${email.trim()}</p>
        <p><strong>Subject:</strong> ${subject?.trim() || 'N/A'}</p>
        <hr>
        <p>${message.trim().replace(/\n/g, '<br>')}</p>
      `,
    });

    return res.json({ ok: true });
  } catch (error) {
    console.error('[contact] send failed:', error);
    return res.status(500).json({ error: 'Failed to send message. Please try again later.' });
  }
});

export default router;
