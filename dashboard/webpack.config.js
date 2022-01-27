var path = require('path'); module.exports = {
  mode: 'development',
  entry: {
          'IndexDataset':  './front/src/EntryPoints/IndexDataset.tsx',
          'IndexModel':  './front/src/EntryPoints/IndexModel.tsx',
         },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
        options:
        {
          "presets": [
            "@babel/preset-env",
            "@babel/preset-react"
          ],
          "plugins": [
            "@babel/plugin-proposal-class-properties",
            "@babel/plugin-syntax-dynamic-import"
          ]
        }
      },
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.(eot|woff|woff2|svg|png|ttf)([\?]?.*)$/, 
        loader: "file-loader",
        options: {
          outputPath: '/'
        }
      },
      {
        test: /\.ts(x?)$/,
        loader: "ts-loader",
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: ['*', '.js', '.jsx', '.tsx']
  },
  output: {
    path: path.resolve(__dirname, 'static'),
    filename: '[name].js'
  },
  devServer: {
    contentBase: './dist'
  }
};
