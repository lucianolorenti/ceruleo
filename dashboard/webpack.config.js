const path = require("path");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
module.exports = {
  mode: 'development',
  entry: {
          'IndexDataset':  './front/src/EntryPoints/IndexDataset.tsx',
          'IndexModel':  './front/src/EntryPoints/IndexModel.tsx',
         },
         plugins: [
          new MiniCssExtractPlugin(),
      
          
        ],
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
        test: /\.scss$/,
        use: [
          MiniCssExtractPlugin.loader,
          { loader: "css-modules-typescript-loader" },
          { loader: "css-loader", options: { modules: true } },

          "sass-loader",
        ],
        sideEffects: true,
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
