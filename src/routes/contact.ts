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

router.get('/contact', (_req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contact Us — Cozmic Astrology</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#0B0E14;color:#F8F9FA;line-height:1.7;-webkit-font-smoothing:antialiased}
.container{max-width:560px;margin:0 auto;padding:40px 20px}
.header{text-align:center;margin-bottom:32px}
h1{font-size:24px;font-weight:700;color:#D4AF37;letter-spacing:.02em}
.subtitle{font-size:14px;color:#A0A0A0;margin-top:4px}
.card{background:rgba(26,27,58,.55);border:1px solid rgba(212,175,55,.15);border-radius:16px;padding:32px}
.form-group{margin-bottom:20px}
label{display:block;font-size:13px;font-weight:600;color:#A0A0A0;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
input,textarea{width:100%;padding:12px 16px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:12px;color:#F8F9FA;font-size:15px;font-family:inherit;outline:none;transition:border-color .2s}
input:focus,textarea:focus{border-color:#D4AF37}
textarea{min-height:120px;resize:vertical}
button{width:100%;padding:14px;background:linear-gradient(135deg,#D4AF37,#B8962F);border:none;border-radius:999px;color:#0B0E14;font-size:16px;font-weight:700;cursor:pointer;transition:opacity .2s,transform .1s}
button:hover{opacity:.9}
button:active{transform:scale(.98)}
button:disabled{opacity:.5;cursor:not-allowed}
.message{padding:12px 16px;border-radius:12px;margin-bottom:16px;font-size:14px;display:none}
.message.success{background:rgba(212,175,55,.12);border:1px solid rgba(212,175,55,.3);color:#D4AF37;display:block}
.message.error{background:rgba(255,68,68,.12);border:1px solid rgba(255,68,68,.3);color:#FF6B6B;display:block}
.required{color:#FF6B6B}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>Contact Us</h1><div class="subtitle">We'd love to hear from you</div></div>
<div class="card">
<div id="status-message" class="message"></div>
<form id="contact-form">
<div class="form-group"><label>Name <span class="required">*</span></label><input type="text" id="name" required placeholder="Your name"></div>
<div class="form-group"><label>Email <span class="required">*</span></label><input type="email" id="email" required placeholder="your@email.com"></div>
<div class="form-group"><label>Subject</label><input type="text" id="subject" placeholder="What's this about?"></div>
<div class="form-group"><label>Message <span class="required">*</span></label><textarea id="message" required placeholder="Tell us what's on your mind..."></textarea></div>
<button type="submit" id="submit-btn">Send Message</button>
</form>
</div>
</div>
<script>
const form=document.getElementById('contact-form'),btn=document.getElementById('submit-btn'),msg=document.getElementById('status-message');
form.addEventListener('submit',async e=>{e.preventDefault();btn.disabled=true;btn.textContent='Sending...';msg.className='message';msg.style.display='none';
try{const r=await fetch('/v1/contact',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:document.getElementById('name').value.trim(),email:document.getElementById('email').value.trim(),subject:document.getElementById('subject').value.trim(),message:document.getElementById('message').value.trim()})});const d=await r.json();if(r.ok){msg.className='message success';msg.textContent="Message sent! We'll get back to you soon.";form.reset()}else{msg.className='message error';msg.textContent=d.error||'Something went wrong.'}}catch(e){msg.className='message error';msg.textContent='Could not reach the server.'}
msg.style.display='block';btn.disabled=false;btn.textContent='Send Message'});
</script>
</body>
</html>`);
});

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
