package globalquake.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.JFrame;
import javax.swing.JPanel;

import globalquake.core.GlobalQuake;
import globalquake.main.Main;
import globalquake.ui.settings.SettingsFrame;

public class GlobalQuakeFrame extends JFrame {

	private final GlobalQuake globalQuake;
	private static final int FPS = 20;

	private boolean hideList = false;
	private final EarthquakeListPanel list;
	private final GlobalQuakePanelOld panel;
	private final JPanel mainPanel;
	private boolean _containsListToggle;
	private boolean _containsSettings;

	public GlobalQuakeFrame(GlobalQuake globalQuake) {
		this.globalQuake = globalQuake;
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		panel = new GlobalQuakePanelOld(globalQuake, this) {

			@Override
			public void paint(Graphics gr) {
				super.paint(gr);
				Graphics2D g = (Graphics2D) gr;
				g.setColor(_containsListToggle ? Color.gray : Color.lightGray);
				g.fillRect(getWidth() - 20, 0, 20, 30);
				g.setColor(Color.black);
				g.drawRect(getWidth() - 20, 0, 20, 30);
				g.setFont(new Font("Calibri", Font.BOLD, 16));
				g.setColor(Color.black);
				g.drawString(hideList ? "<" : ">", getWidth() - 16, 20);

				g.setColor(_containsSettings ? Color.gray : Color.lightGray);
				g.fillRect(getWidth() - 20, getHeight() - 30, 20, 30);
				g.setColor(Color.black);
				g.drawRect(getWidth() - 20, getHeight() - 30, 20, 30);
				g.setFont(new Font("Calibri", Font.BOLD, 16));
				g.setColor(Color.black);
				g.drawString("S", getWidth() - 15, getHeight() - 8);
			}
		};
		panel.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				if (x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30) {
					toggleList();
				}
				if (x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= panel.getHeight() - 30 && y <= panel.getHeight()) {
					SettingsFrame.show();
				}
			}

			@Override
			public void mouseExited(MouseEvent e) {
				_containsListToggle = false;
				_containsSettings = false;
			}
		});
		panel.addMouseMotionListener(new MouseAdapter() {

			@Override
			public void mouseMoved(MouseEvent e) {
				int x = e.getX();
				int y = e.getY();
				_containsListToggle = x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= 0 && y <= 30;
				_containsSettings = x >= panel.getWidth() - 20 && x <= panel.getWidth() && y >= panel.getHeight() - 30
						&& y <= panel.getHeight();
			}
		});

		list = new EarthquakeListPanel(globalQuake);
		panel.setPreferredSize(new Dimension(600, 600));
		list.setPreferredSize(new Dimension(300, 600));

		mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		mainPanel.setPreferredSize(new Dimension(800, 600));
		mainPanel.add(panel, BorderLayout.CENTER);
		mainPanel.add(list, BorderLayout.EAST);

		setContentPane(mainPanel);

		pack();
		setLocationRelativeTo(null);
		setMinimumSize(new Dimension(610, 500));
		setResizable(true);
		setTitle(Main.fullName);

		Timer timer = new Timer();
		timer.scheduleAtFixedRate(new TimerTask() {
			public void run() {
				mainPanel.repaint();
			}
		}, 0, 1000 / FPS);
	}

	protected void toggleList() {
		hideList = !hideList;
		if (hideList) {
			panel.setSize(new Dimension(mainPanel.getWidth(), mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(0, (int) list.getPreferredSize().getHeight()));
		} else {
			panel.setSize(new Dimension(mainPanel.getWidth() - 300, mainPanel.getHeight()));
			list.setPreferredSize(new Dimension(300, (int) list.getPreferredSize().getHeight()));
		}
		_containsListToggle = false;
		_containsSettings = false;
		revalidate();
	}

	public GlobalQuake getGlobalQuake() {
		return globalQuake;
	}

}
