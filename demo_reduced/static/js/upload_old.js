const imagesDir = 'static/uploads';

function uploadFile() {
	let tableTh = $('#dataFile thead tr');
	let tableTd = $('#dataFile tbody');
	$('#fileUp').change(function() {
		tableTh.empty();
		tableTd.empty();
		let dataFile = $(this)[0].files;

		console.log(dataFile);

		let dataImport = []

		$.each(dataFile, function(key, value) {
			dataImport.push({
				name: value.name,
				lastModified: value.lastModified,
				lastModifiedDate: value.lastModifiedDate,
				size: value.size,
				type: value.type,
				webkitRelativePath: value.webkitRelativePath
			});
		});



		console.log(dataImport);

		let th = '';
		let td = '';
		$.each(dataImport[0], function(key, value) {
			th += '<th>' + key + '</th>';
		});

		$.each(dataImport, function(key, value) {
			console.log(value);
			td += '<tr>';
			td += '<td>' + value.name + '</td>';
			td += '<td>' + value.lastModified + '</td>';
			td += '<td>' + value.lastModifiedDate + '</td>';
			td += '<td>' + value.size + '</td>';
			td += '<td>' + value.type + '</td>';
			td += '<td>' + value.webkitRelativePath + '</td>';
			td += '</tr>';
		});

		tableTh.append(th);
		tableTd.append(td);

	});

	$('#but_upload').click(function(event) {
		event.preventDefault();
		let formData = new FormData();

		let dataFiles = $('#fileUp').prop('files')
		console.log(dataFiles);

		$.each(dataFiles, function(i, file) {
			formData.append('file', file);
		});

		console.log(...formData);
		let upImage = new Promise((resolve, reject) => {
			$.ajax({
				url: '/upfile',
				type: 'POST',
				data: formData,
				processData: false,
				contentType: false,
				success: function(data) {
					console.log(data);

				},
				error: function(xhr, status, error) {
					console.log(error);
				}
			});

		});

		let getPrediction = new Promise((resolve, reject) => {
			$('.ecgBox').remove();
			$('.upPrediction').spin();
			setTimeout(function() {
				$.ajax({
					url: '/get_upload_prediction',
					type: 'GET',
					success: function(res) {
						$('.spinner').remove();
						console.log(res.img_path)
						let imgSrc = res.img_path;
						let bubble = '';
						bubble += '<div class="ecgBox">';
						bubble += '<img src=' + imgSrc + ' alt=' + imgSrc + ' />';
						bubble += '<p>' + res.prediction + '</p>';
						bubble += '</div>';

						setTimeout(function() {
							$('.upPrediction').append(bubble);
							$('.ecgBox').fadeIn();
						}, 250);
				},
				error: function(err) {
					console.log(err);
				}
				});
			}, 2000);
		});

		Promise.all([upImage, getPrediction]).then(values => {
			console.log('prediction complete');
		}, reason => {
			console.log('prediction failed: ' + reason);
		});
	});
}

$(window).on('load', function() {
});

$(window).resize(function() {
});

$(document).ready(function() {
	uploadFile();

});
