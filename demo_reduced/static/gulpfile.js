'use strict';
const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const rename = require('gulp-rename');
const cssmin = require('gulp-cssnano');
const prefix = require('gulp-autoprefixer');
const plumber = require('gulp-plumber');
const notify = require('gulp-notify');
const uglify = require('gulp-uglify');
const sourcemaps = require('gulp-sourcemaps');
const browserSync = require('browser-sync').create();
const concat = require('gulp-concat');
const gulpMultiProcess = require('gulp-multi-process');
const merge = require('merge-stream');
const cache = require('gulp-cache');

const paths = {
	sass_src: 'sass/',
	css_build: 'css/',
	js_sing: 'js/',
	js_src: 'lib/',
	js_build: 'vendors/',
	htmls: '../templates/'
};

const projectURL = '127.0.0.1:8000';

const displayError = function(error) {
	// Initial building up of the error
	var errorString = '[' + error.plugin.error.bold + ']';
	errorString += ' ' + error.message.replace('\n', ''); // Removes new line at the end

	// If the error contains the filename or line number add it to the string
	if (error.fileName)
		errorString += ' in ' + error.fileName;

	if (error.lineNumber)
		errorString += ' on line ' + error.lineNumber.bold;

	// This will output an error like the following:
	// [gulp-sass] error message in file_name on line 1
	console.error(errorString);
};

const onError = function(err) {
	notify.onError({
		title: 'Gulp',
		subtitle: 'Failure!',
		message: 'Error: <%= error.message %>',
		sound: 'Basso'
	})(err);
	this.emit('end');
};

const sassOptions = {
	outputStyle: 'expanded'
};

const prefixerOptions = {
	browsers: ['last 2 versions']
};

function reload(done) {
	browserSync.reload();
	done();
}

function clear(done) {
	return cache.clearAll(done);
}

function watch() {
	gulp.watch(paths.sass_src + '**/*.scss', gulp.series(styles, clear, reload));
	gulp.watch(paths.js_sing + '*.js', gulp.series(clear, reload));
	gulp.watch(paths.htmls + '*.html', gulp.series(clear, reload));
	gulp.watch(paths.js_src + '*.js', gulp.series(vendJs, clear, reload));
}

function serve(done) {
	browserSync.init({
		proxy: projectURL,
		open: true,
		injectChanges: true,
		stream: true
	});
	done();
}

function styles() {
	return gulp.src(paths.sass_src + 'index.scss').pipe(plumber({errorHandler: onError})).pipe(sourcemaps.init({largeFile: true})).pipe(sass(sassOptions)).pipe(prefix(prefixerOptions)).pipe(rename('index.css')).pipe(sourcemaps.write(paths.sass_src, {
		mapFile: function(mapFilePath) {
			return mapFilePath.replace('.min.css.map', '.map');
		}
	})).pipe(gulp.dest(paths.css_build));
	// .pipe(cssmin())
	// .pipe(rename({ suffix: '.min' }))
	// .pipe(gulp.dest(paths.css_build)
}

function vendJs() {
	const vendors = gulp.src([
		paths.js_src + 'jquery.min.js',
		paths.js_src + 'jquery-migrate.min.js',
		paths.js_src + 'spin.min.js',
		paths.js_src + 'jquery-ext.js'
	]).pipe(plumber()).pipe(uglify()).pipe(concat('vendors.min.js')).pipe(gulp.dest(paths.js_build));

	return merge(vendors);
}

function multi(done) {
	return gulpMultiProcess(['styles', 'vendJs'], done);
}

exports.styles = styles;
exports.vendJs = vendJs;
exports.watch = watch;
exports.reload = reload;
exports.clear = clear;
exports.multi = multi;

gulp.task(
	'default',
	gulp.series(multi, serve, watch)
);
