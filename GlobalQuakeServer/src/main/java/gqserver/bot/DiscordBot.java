package gqserver.bot;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeReportEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScales;
import globalquake.core.intensity.Level;
import globalquake.core.intensity.MMIIntensityScale;
import globalquake.utils.GeoUtils;
import net.dv8tion.jda.api.EmbedBuilder;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.events.session.ReadyEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.requests.GatewayIntent;

import javax.imageio.ImageIO;
import javax.security.auth.login.LoginException;

import globalquake.core.Settings;
import net.dv8tion.jda.api.utils.FileUpload;
import org.tinylog.Logger;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Instant;

public class DiscordBot extends ListenerAdapter{

    private static JDA jda;

    public static void init() throws LoginException {
        jda = JDABuilder.createDefault(Settings.discordBotToken).enableIntents(GatewayIntent.GUILD_MESSAGES)
                .addEventListeners(new DiscordBot())
                .build();

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                sendQuakeCreateInfo(event.earthquake());
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                sendQuakeUpdateInfo(event.earthquake());
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                sendQuakeRemoveInfo(event.earthquake());
            }

            @Override
            public void onQuakeReport(QuakeReportEvent event) {
                sendQuakeReportInfo(event);
            }
        });
    }

    private static void sendQuakeReportInfo(QuakeReportEvent event) {
        if(jda == null){
            return;
        }

        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        if(channel == null){
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("Final Report");

        builder.setImage("attachment://map.png");
        builder.setThumbnail("attachment://int.png");
        createDescription(builder, event.earthquake());

        ByteArrayOutputStream baosMap = new ByteArrayOutputStream();
        ByteArrayOutputStream baosInt = new ByteArrayOutputStream();
        try {
            ImageIO.write(event.map(), "png", baosMap);
            ImageIO.write(event.intensities(), "png", baosInt);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        channel.sendMessageEmbeds(builder.build())
                .addFiles(FileUpload.fromData(baosMap.toByteArray(), "map.png"), FileUpload.fromData(baosInt.toByteArray(), "int.png"))
                .queue();

    }

    private static void sendQuakeRemoveInfo(Earthquake earthquake) {
        if(jda == null){
            return;
        }

        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);

        if(channel == null){
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("M%.1f %s".formatted(earthquake.getMag(), earthquake.getRegion()));
        builder.setAuthor("CANCELED");

        channel.sendMessageEmbeds(builder.build()).queue();
    }

    private static void sendQuakeUpdateInfo(Earthquake earthquake) {
        if(true){
            return;
        }
        if(jda == null){
            return;
        }

        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        if(channel == null){
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("Revision #%d".formatted(earthquake.getRevisionID()));
        createDescription(builder, earthquake);

        channel.sendMessageEmbeds(builder.build()).queue();
    }

    private static void sendQuakeCreateInfo(Earthquake earthquake) {
        if(jda == null){
            return;
        }

        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        if(channel == null){
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setAuthor("New Event");
        createDescription(builder, earthquake);

        channel.sendMessageEmbeds(builder.build()).queue();
    }

    private static void createDescription(EmbedBuilder builder, Earthquake earthquake) {
        builder.setTitle("M%.1f %s".formatted(
                earthquake.getMag(),
                earthquake.getRegion()));

        builder.setDescription(
                "Depth: %.1fkm / %.1fmi\n".formatted(earthquake.getDepth(), earthquake.getDepth() * DistanceUnit.MI.getKmRatio()) +
                "Time: %s\n".formatted(Settings.formatDateTime(Instant.ofEpochMilli(earthquake.getOrigin()))) +
                "Quality: %s (%d stations)".formatted(earthquake.getCluster().getPreviousHypocenter().quality.getSummary(), earthquake.getCluster().getAssignedEvents().size())
        );

        double pga = GeoUtils.pgaFunction(earthquake.getMag(), 0, earthquake.getDepth());

        Level level = IntensityScales.getIntensityScale().getLevel(pga);
        Color levelColor = level == null ? Color.gray : level.getColor();

        builder.setColor(levelColor);
        builder.setFooter("Created at: %s".formatted(Settings.formatDateTime(Instant.now())));
    }

    @Override
    public void onReady(ReadyEvent event) {
        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        if(channel == null){
            return;
        }

        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("GlobalQuake BOT");
        builder.setDescription("Initialising...");

        channel.sendMessageEmbeds(builder.build()).queue();

        Logger.info("Discord bot ready");
    }

}
