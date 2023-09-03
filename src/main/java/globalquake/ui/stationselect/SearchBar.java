package globalquake.ui.stationselect;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Insets;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.Ellipse2D;

import javax.swing.Timer;

import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.border.EmptyBorder;

import org.json.simple.parser.JSONParser;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

public class SearchBar extends JTextField{
    
    private final Color backgroundColor = Color.WHITE;
    private Color animationColor = new Color(160,160,160);
    private final Icon iconSearch;
    private final Icon iconClose;
    private final Icon iconLoading;
    private final String hintText = "Searching ... ";
    private String search = "";
    private Timer timer;
    private boolean show;
    private float speed = 8f;
    private float location = -1;


    public SearchBar(){
        setBackground((new Color(255,255,255,0)));
        setOpaque(false);
        setBorder(new EmptyBorder(10, 10, 10, 50));
        setFont(new java.awt.Font("Calibri", 0, 14));
        setSelectionColor(new Color(80, 199, 255));
        iconSearch = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/search.png"));
        iconClose = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/close.png"));
        iconLoading = new javax.swing.ImageIcon(getClass().getResource("/image_icons/search_icons/loading.gif"));
        

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent me){
                if(checkMouseOver(getMousePosition())){
                    setCursor(new Cursor(Cursor.HAND_CURSOR));
                }
                else{
                    setCursor(new Cursor(Cursor.TEXT_CURSOR));
                }
            }
        });
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent me){
                if(SwingUtilities.isLeftMouseButton(me)){
                    if(checkMouseOver(me.getPoint())){
                        //Mouse Pressed over the button
                        search = getText();
                        if(!timer.isRunning()){
                            if(show){
                                setEditable(true);
                                show = false;
                                location = -1;
                                timer.start();
                            }
                            else{
                                setEditable(false);
                                show = true;
                                location = getWidth();
                                timer.start();
                            }
                        }
                        parseSearch(search);
                    }
                }
            }
        });
        addKeyListener(new java.awt.event.KeyAdapter() {
            @Override
            public void keyPressed(java.awt.event.KeyEvent evt) {
                StationSelectFrame.suggestionPanel.removeAll();
                StationSelectFrame.suggestionPanel.revalidate();
                String phrase = getText().trim();
                if(phrase.length()>0){
                    StationSelectFrame.suggestionPanel.setVisible(true);
                    String[] phrasesToList = generatePredictedText(phrase);
                    for(int i = 0; i < phrasesToList.length; i++){
                        JButton button = new JButton(phrasesToList[i]);
                        button.setPreferredSize(new Dimension(375, 25));
                        button.setBorderPainted(false);
                        button.setFocusPainted(false);
                        button.setContentAreaFilled(false);
                        button.addMouseListener(new java.awt.event.MouseAdapter() {
                            public void mouseEntered(java.awt.event.MouseEvent evt) {
                                button.setContentAreaFilled(true);
                            }
                            public void mouseExited(java.awt.event.MouseEvent evt) {
                                button.setContentAreaFilled(false);
                            }
                        });
                        button.addActionListener(new java.awt.event.ActionListener() {
                            public void actionPerformed(java.awt.event.ActionEvent evt) {
                                setText(button.getText());
                                StationSelectFrame.suggestionPanel.setVisible(false);
                            }
                        });
                        StationSelectFrame.suggestionPanel.add(button);
                    }
                }
                else{
                    StationSelectFrame.suggestionPanel.setVisible(false);
                }
            }
        });
        timer=new Timer(0, new ActionListener() {
            @Override
            public void actionPerformed(java.awt.event.ActionEvent ae) {
                if(show){
                    if(location>0){
                        location-=speed;
                        repaint();
                    }
                    else{
                        timer.stop();
                    }
                }
                else{
                    if(location<getWidth()){
                        location+=speed;
                        repaint();
                    }
                    else{
                        timer.stop();
                    }
                }
            }
        });

    }

    @Override
    protected void paintComponent(java.awt.Graphics g) {
        int width = getWidth();
        int height = getHeight();
        setPreferredSize(new Dimension(200, 200));
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING, java.awt.RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.setColor(backgroundColor);
        setPreferredSize(new Dimension(width, height));
        g2.fillRoundRect(0, 0, width, height, height, height);
        super.paintComponent(g);
        //Create Button
        int marginButton = 5;
        int buttonSize = height - marginButton * 2;
        GradientPaint gp = new GradientPaint(0, 0, new Color(255,255,255,0), width, 0, animationColor);
        g2.setPaint(gp);
        g2.fillOval((width - height)+3, marginButton, buttonSize, buttonSize);

        //Create Animation
        if(location != -1){
            g2.fillRoundRect((int)location, 0, (int)(width-location), height, height, height);
            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, getAlpha()));
            //Create Loading Icon
            int iconWidth = 30;
            int iconHeight = 30;
            g2.drawImage(((javax.swing.ImageIcon) iconLoading).getImage(), (int)(location-5), (height-iconHeight)/2, iconWidth, iconHeight, this);
        }
        
        //Create Button Icon
        g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f));
        int marginImage=5;
        int imageSize = buttonSize - marginImage * 2;
        Image image;
        if(show){
            image = ((javax.swing.ImageIcon) iconClose).getImage();
            search = "";
            setText("");
        }
        else{
            image = ((javax.swing.ImageIcon) iconSearch).getImage();
        }
        g2.drawImage(image, width-height+marginImage + 3, marginButton+marginImage, imageSize, imageSize, null);
        g2.dispose();
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        if (getText().length() == 0) {
            int h = getHeight();
            ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            Insets ins = getInsets();
            FontMetrics fm = g.getFontMetrics();
            int c0 = getBackground().getRGB();
            int c1 = getForeground().getRGB();
            int m = 0xfefefefe;
            int c2 = ((c0 & m) >>> 1) + ((c1 & m) >>> 1);
            g.setColor(new Color(c2, true));
            g.drawString(hintText, ins.left, h / 2 + fm.getAscent() / 2 - 2);
        }
    }
    private float getAlpha(){
        float width = getWidth()/2;
        float alpha = (location)/(-width);
        alpha+=1;
        if(alpha<0){
            alpha = 0f;
        }
        if(alpha>1){
            alpha = 1f;
        }
        return alpha;
    }
    private boolean checkMouseOver(Point mouse){
        if(mouse==null){
            return false;
        }
        int width = getWidth();
        int height = getHeight();
        int buttonSize = height - 5 *2;
        Point point = new Point(width-height+3, 5);
        Ellipse2D.Double circle = new Ellipse2D.Double(point.x, point.y, buttonSize, buttonSize);
        return circle.contains(mouse);
    }

    private void parseSearch(String search){
        try{
            JSONParser parser = new JSONParser();
            JSONObject rootObject = (JSONObject) parser.parse(new FileReader("src/main/resources/polygons/countriesHD.json"));

            JSONArray countryData = (JSONArray) rootObject.get("features");
           
            String userInput = search.toLowerCase();
            userInput = userInput.substring(0, 1).toUpperCase() + userInput.substring(1);

            for(Object obj: countryData){
                JSONObject countryObject = (JSONObject) obj;
                JSONObject properties = (JSONObject) countryObject.get("properties");
                String countryName = (String) properties.get("name");

                if(countryName.equals(userInput)){
                    System.out.println("Found country: " + countryName);
                }

            }
        }
        catch(Exception e){
            System.out.println("Error: " + e);
        }
    }
    
    private String[] generatePredictedText(String input){
        String[] listOfAllPhrases = getWords("src/main/resources/search_suggestions/countries.txt");

        return generatePhraseList(input, listOfAllPhrases);
    }

    private String[] generatePhraseList(String input, String[] listOfPhrases){
        ArrayList<String> phrases = new ArrayList<String>();

        for(int i = 0; i < listOfPhrases.length; i++){
            if(listOfPhrases[i].toLowerCase().startsWith(input.toLowerCase())){
                phrases.add(listOfPhrases[i]);
            }
        }

        return phrases.toArray(new String[0]);
    }

    private String[] getWords(String filepath){
        ArrayList<String> phrases = new ArrayList<String>();

        try{
            FileReader fileReader = new FileReader(filepath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;
            while((line = bufferedReader.readLine()) != null){
                phrases.add(line);
            }
            bufferedReader.close();
            fileReader.close();

            Collections.sort(phrases);
        }
        catch(Exception e){
            phrases.add(e.toString());
        }

        return phrases.toArray(new String[0]);
    }
}
