package globalquake.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import globalquake.core.Earthquake;
import globalquake.main.Settings;
import globalquake.geo.GeoUtils;
import globalquake.utils.Level;
import globalquake.utils.Shindo;
import globalquake.geo.TravelTimeTable;

public class AlertWindow extends JFrame {

	private static final long serialVersionUID = 1L;
	private JPanel contentPane;
	private JLabel lblMag;
	private JLabel lblDist;
	private JPanel panelP;
	private JLabel lblPSec;
	private JPanel panelS;
	private JLabel lblSSec;
	private JPanel panelIntensity;
	private JLabel lblShindo;
	private Earthquake earthquake;
	private Thread uiThread;
	private Color chillColor = new Color(51, 204, 255);
	private Color strongColor = new Color(255, 204, 51);

	public AlertWindow(Earthquake earthquake) {
		this();
		this.earthquake = earthquake;
		uiThread = new Thread() {
			public void run() {
				while (true) {
					try {
						sleep(200);
					} catch (InterruptedException e) {
						break;
					}
					updateInfo();
				}
			};
		};
		uiThread.start();

		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				uiThread.interrupt();
			}
		});
	}

	public static final DecimalFormat f1d = new DecimalFormat("0.0", new DecimalFormatSymbols(Locale.ENGLISH));

	private void updateInfo() {
		lblMag.setText("M" + f1d.format(earthquake.getMag()) + " Earthquake struck!");

		double distGC = GeoUtils.greatCircleDistance(earthquake.getLat(), earthquake.getLon(), Settings.homeLat,
				Settings.homeLon);
		double distGEO = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(), -earthquake.getDepth(),
				Settings.homeLat, Settings.homeLon, 0.0);

		lblDist.setText(f1d.format(distGC) + "km from home location");

		double age = (System.currentTimeMillis() - earthquake.getOrigin()) / 1000.0;

		if (age > 60 * 30) {
			this.dispose();
			uiThread.interrupt();
			return;
		}

		double pTravel = (long) (TravelTimeTable.getPWaveTravelTime(earthquake.getDepth(),
				TravelTimeTable.toAngle(distGC)));
		double sTravel = (long) (TravelTimeTable.getSWaveTravelTime(earthquake.getDepth(),
				TravelTimeTable.toAngle(distGC)));

		int secondsP = (int) Math.max(0, Math.ceil(pTravel - age));
		int secondsS = (int) Math.max(0, Math.ceil(sTravel - age));

		panelP.setBackground(secondsP > 0 ? new Color(0, 153, 255) : Color.LIGHT_GRAY);
		panelS.setBackground(secondsS > 0 ? new Color(255, 51, 0) : Color.LIGHT_GRAY);

		lblPSec.setText(secondsP + "s");
		lblSSec.setText(secondsS + "s");

		double pga = GeoUtils.pgaFunctionGen1(earthquake.getMag(), distGEO);
		Level shindo = Shindo.getLevel(pga);

		Color c = Color.LIGHT_GRAY;
		String str = "-";

		boolean chill = shindo == null;

		contentPane.setBackground(chill ? chillColor : strongColor);

		if (!chill) {
			c = Shindo.getColorShindo(shindo);
			str = shindo.getName();
		}

		lblShindo.setText(str);
		panelIntensity.setBackground(c);
	}

	public AlertWindow() {
		setTitle("Warning!");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setResizable(false);
		contentPane = new JPanel();
		// contentPane.setBackground();
		contentPane.setPreferredSize(new Dimension(400, 220));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		JLabel lblNewLabel = new JLabel("Earthquake!!!");
		lblNewLabel.setHorizontalAlignment(SwingConstants.CENTER);
		lblNewLabel.setFont(new Font("Dialog", Font.BOLD, 32));
		lblNewLabel.setForeground(Color.RED);
		lblNewLabel.setBounds(12, 12, 376, 40);
		contentPane.add(lblNewLabel);

		lblMag = new JLabel("M?.? Earthquake struck");
		lblMag.setHorizontalAlignment(SwingConstants.CENTER);
		lblMag.setFont(new Font("Dialog", Font.BOLD, 24));
		lblMag.setBounds(22, 60, 356, 40);
		contentPane.add(lblMag);

		lblDist = new JLabel("???km from home location");
		lblDist.setHorizontalAlignment(SwingConstants.CENTER);
		lblDist.setFont(new Font("Dialog", Font.BOLD, 20));
		lblDist.setBounds(22, 98, 366, 40);
		contentPane.add(lblDist);

		panelP = new JPanel();
		panelP.setBackground(new Color(0, 153, 255));
		panelP.setBounds(22, 145, 98, 63);
		contentPane.add(panelP);
		panelP.setLayout(new BorderLayout(0, 0));

		lblPSec = new JLabel("???s");
		lblPSec.setFont(new Font("Dialog", Font.BOLD, 30));
		lblPSec.setHorizontalAlignment(SwingConstants.CENTER);
		panelP.add(lblPSec);

		panelS = new JPanel();
		panelS.setBackground(new Color(255, 51, 0));
		panelS.setBounds(290, 145, 98, 63);
		contentPane.add(panelS);
		panelS.setLayout(new BorderLayout(0, 0));

		lblSSec = new JLabel("???s");
		lblSSec.setHorizontalAlignment(SwingConstants.CENTER);
		lblSSec.setFont(new Font("Dialog", Font.BOLD, 30));
		panelS.add(lblSSec, BorderLayout.CENTER);

		panelIntensity = new JPanel();
		panelIntensity.setBackground(new Color(102, 255, 153));
		panelIntensity.setBounds(132, 145, 146, 63);
		contentPane.add(panelIntensity);
		panelIntensity.setLayout(new BorderLayout(0, 0));

		lblShindo = new JLabel("??");
		lblShindo.setFont(new Font("Dialog", Font.BOLD, 36));
		lblShindo.setHorizontalAlignment(SwingConstants.CENTER);
		panelIntensity.add(lblShindo, BorderLayout.CENTER);

		pack();
		setLocationRelativeTo(null);
	}
}
