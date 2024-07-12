const express = require('express');
const fileUpload = require('express-fileupload');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const app = express();
const port = 3000;

// Enable file upload
app.use(fileUpload());

// Serve the HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Handle form submission
app.post('/generate', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send('No image uploaded');
  }

  const image = req.files.image;
  const formData = new FormData();
  formData.append('image', image.data, image.name);
  formData.append('remove_background', req.body.remove_background);
  formData.append('sample_steps', req.body.sample_steps);
  formData.append('sample_seed', req.body.sample_seed);

  try {
    const response = await axios.post('http://localhost:5000/generate', formData, {
      headers: formData.getHeaders(),
    });

    res.json(response.data);
  } catch (error) {
    res.status(500).send(error.message);
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
